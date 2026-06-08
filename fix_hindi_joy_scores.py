import argparse
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from emotion_bootstrapper import VerboseSemanticBootstrapper

def fix_hindi_joy_scores(
    input_parquet_path: str,
    output_parquet_path: str,
    model_name: str | None = None,
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
        device_map: Device to run the model on (e.g., "auto", "cpu", "cuda:0").
        inference_batch_size: Batch size for model inference.
    """
    print(f"Loading dataset from {input_parquet_path}...")
    df = pd.read_parquet(input_parquet_path)
    print(f"Dataset loaded with {len(df)} rows.")

    if "translation_language" not in df.columns:
        raise ValueError(
            "The input dataset must contain a 'translation_language' column "
            "to identify Hindi entries."
        )
    if "text" not in df.columns:
        raise ValueError("The input dataset must contain a 'text' column.")
    if "emotion_vector" not in df.columns:
        print(
            "Warning: 'emotion_vector' column not found. This script will add it "
            "for Hindi entries, but other entries will remain without emotion vectors."
        )

    hindi_mask = df["translation_language"].str.lower() == "hindi"
    hindi_df = df[hindi_mask].copy()

    if hindi_df.empty:
        print("No Hindi entries found in the dataset. No changes needed.")
        df.to_parquet(output_parquet_path, index=False)
        return

    print(f"Found {len(hindi_df)} Hindi entries to re-bootstrap.")

    # Initialize the bootstrapper with multilingual support
    # It will now use the corrected SEMANTIC_HYPOTHESES_MULTILINGUAL from emotion_bootstrapper.py
    bootstrapper = VerboseSemanticBootstrapper(
        model=model_name, device_map=device_map, multilingual=True
    )

    hindi_texts = hindi_df["text"].tolist()
    print("Re-bootstrapping emotion scores for Hindi entries...")
    new_emotion_vectors = bootstrapper.label_texts(
        hindi_texts, inference_batch_size=inference_batch_size
    )

    hindi_df["emotion_vector"] = new_emotion_vectors

    # Update the original DataFrame with the corrected Hindi entries
    df.loc[hindi_mask, "emotion_vector"] = hindi_df["emotion_vector"]

    print(f"Saving updated dataset to {output_parquet_path}...")
    df.to_parquet(output_parquet_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Hindi 'joy' emotion scores in an augmented dataset.")
    parser.add_argument("--input-parquet", required=True, help="Path to the input augmented Parquet file.")
    parser.add_argument("--output-parquet", required=True, help="Path to save the corrected Parquet file.")
    parser.add_argument("--model", default=None, help="Hugging Face model name (default: mDeBERTa-v3-base-mnli-xnli).")
    parser.add_argument("--device-map", default="auto", help="Device to run the model on (e.g., 'auto', 'cpu', 'cuda:0').")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference.")
    args = parser.parse_args()

    fix_hindi_joy_scores(
        args.input_parquet,
        args.output_parquet,
        args.model,
        args.device_map,
        args.batch_size,
    )