# %%
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet"
)

label_distribution = df["label"].value_counts(normalize=True)
# %%
avg_distribution = label_distribution.mean()
print(avg_distribution)
# %%
target = int(len(df) * 0.35)
labels = df["label"].unique()
k = len(labels)
quota = target // k
samples = []

for label in labels:
    group = df[df["label"] == label]
    take = min(len(group), quota)
    samples.append(group.sample(n=take, random_state=42))

# concatenate what we have
partial = pd.concat(samples)

# Calculate remaining samples needed
remaining = target - len(partial)

# sample remaining from the leftover pool
leftover_pool = df.drop(partial.index)
fill = leftover_pool.sample(n=remaining, random_state=42)

sample_df = pd.concat([partial, fill]).sample(frac=1, random_state=42)
print(len(sample_df))
sample_df["label"].value_counts(normalize=True)

# %%
# drop the index
sample_df = sample_df.reset_index(drop=True)
sample_df.to_parquet(
    "sampled_emotions.parquet"
)  # DerivedFunction01/dair-ai_emotions_sample
