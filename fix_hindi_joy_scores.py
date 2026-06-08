import argparse
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from datasets import load_dataset # Added import for load_dataset

from emotion_bootstrapper import VerboseSemanticBootstrapper


def fix_hindi_joy_scores(
    input_parquet_path: str,
    output_parquet_path: str,
    model_name: str | None = None,
    source_multilingual_dataset_path: str, # Added argument
    source_text_column: str = "text", # Added argument
    source_language_column: str = "translation_language", # Added argument
    device_map: str = "auto",
    inference_batch_size: int = 32,
) -> None:
    """
    Loads an augmented dataset, identifies Hindi entries, re-bootstraps their
    emotion scores using the corrected Hindi 'joy' hypothesis, and saves the
    updated dataset.

    Args:
        input_parquet_path: Path to the input augmented Parquet file.
        output_parquet_path: Path to save the corrected Parquet file.
        model_name: The name of the Hugging Face model to use for bootstrapping.
                    Defaults to "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" for multilingual.
        source_multilingual_dataset_path: Path to the Hugging Face dataset
                                          containing the original multilingual texts.
        source_text_column: The name of the text column in the source multilingual dataset.
        source_language_column: The name of the language column in the source multilingual dataset.
        device_map: Device to run the model on (e.g., "auto", "cpu", "cuda:0").
        inference_batch_size: Batch size for model inference.
    """
    print(f"Loading target dataset from {input_parquet_path}...")
    df_target = pd.read_parquet(input_parquet_path)
    print(f"Target dataset loaded with {len(df_target)} rows.")

    if "text" not in df_target.columns:
        raise ValueError("The target input dataset must contain a 'text' column.")
    if "emotion_vector" not in df_target.columns:
        raise ValueError(
            "The target input dataset must contain an 'emotion_vector' column."
        )

    print(f"Loading source multilingual dataset from {source_multilingual_dataset_path}...")
    # Load the source dataset (e.g., DerivedFunction01/mt-emotions)
    # Assuming it's a Hugging Face dataset, we load the 'train' split or the first available.
    source_dataset_dict = load_dataset(source_multilingual_dataset_path)
    source_dataset = (
        source_dataset_dict["train"]
        if "train" in source_dataset_dict
        else next(iter(source_dataset_dict.values()))
    )
    print(f"Source dataset loaded with {len(source_dataset)} rows.")

    if source_text_column not in source_dataset.column_names:
        raise ValueError(
            f"The source dataset must contain a '{source_text_column}' column."
        )
    if source_language_column not in source_dataset.column_names:
        raise ValueError(
            f"The source dataset must contain a '{source_language_column}' column "
            "to identify Hindi entries."
        )

    # Filter source dataset for Hindi entries
    hindi_source_df = source_dataset.to_pandas()
    hindi_source_mask = hindi_source_df[source_language_column].str.lower() == "hindi"
    hindi_texts_to_rebootstrap = hindi_source_df.loc[hindi_source_mask, source_text_column].tolist()

    if not hindi_texts_to_rebootstrap:
        print("No Hindi entries found in the source dataset. No changes needed.")
        df_target.to_parquet(output_parquet_path, index=False)
        return

    print(f"Found {len(hindi_texts_to_rebootstrap)} Hindi texts in the source dataset to re-bootstrap.")

    # Initialize the bootstrapper with multilingual support
    # It will now use the corrected SEMANTIC_HYPOTHESES_MULTILINGUAL from emotion_bootstrapper.py
    bootstrapper = VerboseSemanticBootstrapper(
        model=model_name, device_map=device_map, multilingual=True
    )

    print("Re-bootstrapping emotion scores for Hindi texts...")
    new_emotion_vectors_for_hindi = bootstrapper.label_texts(
        hindi_texts_to_rebootstrap, inference_batch_size=inference_batch_size
    )

    # Create a mapping from Hindi text to its new 'joy' score
    hindi_text_to_new_joy_score = {
        text: vector.get("joy")
        for text, vector in zip(hindi_texts_to_rebootstrap, new_emotion_vectors_for_hindi)
        if "joy" in vector # Ensure 'joy' is present
    }

    print("Updating 'joy' scores in the target dataset by text matching...")
    # Iterate through the target DataFrame and update 'joy' scores
    updated_emotion_vectors = df_target["emotion_vector"].tolist()
    texts_in_target = df_target["text"].tolist()

    num_updated = 0
    for i, text in enumerate(tqdm(texts_in_target, desc="Matching and updating joy scores")):
        if text in hindi_text_to_new_joy_score:
            new_joy_score = hindi_text_to_new_joy_score[text]
            if updated_emotion_vectors[i] is None:
                updated_emotion_vectors[i] = {} # Initialize if None
            updated_emotion_vectors[i]["joy"] = new_joy_score
            num_updated += 1

    df_target["emotion_vector"] = updated_emotion_vectors

    print(f"Successfully updated 'joy' scores for {num_updated} Hindi entries in the target dataset.")
    print(f"Saving updated dataset to {output_parquet_path}...")
    df_target.to_parquet(output_parquet_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix Hindi 'joy' emotion scores in an augmented dataset by text matching."
    )
    parser.add_argument(
        "--input-parquet",
        required=True,
        help="Path to the input augmented Parquet file.",
    )
    parser.add_argument(
        "--output-parquet",
        required=True,
        help="Path to save the corrected Parquet file.",
    )
    parser.add_argument(
        "--source-multilingual-dataset-path",
        required=True,
        help="Path to the Hugging Face dataset containing the original multilingual texts (e.g., DerivedFunction01/mt-emotions).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model name (default: mDeBERTa-v3-base-mnli-xnli).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device to run the model on (e.g., 'auto', 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for model inference."
    )
    parser.add_argument(
        "--source-text-column",
        default="text",
        help="Name of the text column in the source multilingual dataset.",
    )
    parser.add_argument(
        "--source-language-column",
        default="translation_language",
        help="Name of the language column in the source multilingual dataset (e.g., 'translation_language' or 'lang').",
    )
    args = parser.parse_args()

    fix_hindi_joy_scores(
        args.input_parquet,
        args.output_parquet,
        args.model,
        args.device_map,
        args.batch_size,
        args.source_multilingual_dataset_path,
        args.source_text_column,
        args.source_language_column,
    )
