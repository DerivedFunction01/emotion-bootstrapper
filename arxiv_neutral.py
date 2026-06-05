# %%
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.notebook import tqdm
import pysbd
import multiprocessing as mp

df = pd.read_parquet(
    "hf://datasets/DerivedFunction01/arxiv_sample/arxiv_sample.parquet"
)
df.head()
# %%
abs_df = df[["abstract"]]
abs_df.head()

# %%

# 1. Define a helper function to process a chunk of text
# (pysbd segmenter is initialized inside the worker to avoid serialization issues)
def process_chunk(text_series):
    # Ensure it's a pandas Series so we can use .apply()
    if not isinstance(text_series, pd.Series):
        text_series = pd.Series(text_series)

    segmenter = pysbd.Segmenter(language="en", clean=False)

    def get_first_n_sentences(text, n=2):
        if not isinstance(text, str):
            return ""
        return " ".join(segmenter.segment(text)[:n])

    return text_series.apply(get_first_n_sentences)


# 2. Split the dataframe into chunks based on available CPUs
num_cores = mp.cpu_count()
chunks = np.array_split(abs_df["abstract"], num_cores)

# 3. Process chunks in parallel using ProcessPoolExecutor
results = []
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Submit all chunks to the pool
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

    # Use tqdm to monitor progress as chunks finish
    for future in tqdm(futures, desc="Processing chunks"):
        results.extend(future.result())
# %%
# 4. Combine the results back into the dataframe
# Combine the chunked Series back into a single Series
text_df = pd.DataFrame(results)
# Set the column name to text
text_df.columns = ["text"]
text_df.head()
# %% export text_df
text_df.to_parquet("arxiv_sample_neutral.parquet")

# On hf: DerivedFunction1/arxiv_sample_first_sentences

# %%
