import pandas as pd
import numpy as np

# ------------------------------------------------
# Paths
# ------------------------------------------------
input_csv = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_metadata_supervised.csv"

train_out = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_split.csv"
val_out   = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/val_split.csv"

# ------------------------------------------------
# Load
# ------------------------------------------------
df = pd.read_csv(input_csv)

# ------------------------------------------------
# Extract identity (idXXXX)
# ------------------------------------------------
df["identity"] = df["path"].apply(lambda x: x.split("/")[0])

unique_ids = df["identity"].unique()
print("Total identities:", len(unique_ids))

# ------------------------------------------------
# Shuffle identities (reproducible)
# ------------------------------------------------
np.random.seed(42)
np.random.shuffle(unique_ids)

half = len(unique_ids) // 2
train_ids = set(unique_ids[:half])
val_ids   = set(unique_ids[half:])

# ------------------------------------------------
# Split
# ------------------------------------------------
train_df = df[df["identity"].isin(train_ids)].copy()
val_df   = df[df["identity"].isin(val_ids)].copy()

# Remove helper column
train_df.drop(columns=["identity"], inplace=True)
val_df.drop(columns=["identity"], inplace=True)

# ------------------------------------------------
# Sanity Checks
# ------------------------------------------------
print("\nTrain samples:", len(train_df))
print("Val samples:", len(val_df))

print("\nTrain identities:", train_df["path"].apply(lambda x: x.split("/")[0]).nunique())
print("Val identities:", val_df["path"].apply(lambda x: x.split("/")[0]).nunique())

# Check overlap
train_ids_check = set(train_df["path"].apply(lambda x: x.split("/")[0]))
val_ids_check   = set(val_df["path"].apply(lambda x: x.split("/")[0]))

print("Identity overlap:", len(train_ids_check & val_ids_check))

# ------------------------------------------------
# Save
# ------------------------------------------------
train_df.to_csv(train_out, index=False)
val_df.to_csv(val_out, index=False)

print("\nSaved:")
print("Train →", train_out)
print("Val   →", val_out)