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
unseen_pool = df.drop(sample_df.index)
sample_df = sample_df.reset_index(drop=True)
sample_df.to_parquet(
    "sampled_emotions.parquet"
)  # DerivedFunction01/dair-ai_emotions_sample
# %%
# Define your target size for this new training run
target = int(len(df) * 0.25)
labels = unseen_pool["label"].unique()
k = len(labels)
quota = target // k
samples = []

# Execute your exact stratified quota sampling, but strictly inside the unseen pool
for label in labels:
    group = unseen_pool[unseen_pool["label"] == label]
    take = min(len(group), quota)
    samples.append(group.sample(n=take, random_state=42))

# Concatenate the baseline quota allocations
partial = pd.concat(samples)

# Calculate remaining samples needed to hit the target
remaining = target - len(partial)

# Sample the rest from the leftover rows inside the unseen pool only
leftover_unseen = unseen_pool.drop(partial.index)
fill = leftover_unseen.sample(n=remaining, random_state=42)

# Combine, shuffle, and reset index
new_sample_df = pd.concat([partial, fill]).sample(frac=1, random_state=42)
new_sample_df = new_sample_df.reset_index(drop=True)

print(f"New Unseen Dataset Size: {len(new_sample_df)}")
print(new_sample_df["label"].value_counts(normalize=True))

# %%
# Save the new training slice separately
new_sample_df.to_parquet("sampled_emotions2.parquet")

# %%
