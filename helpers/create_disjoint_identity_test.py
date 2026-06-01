import os
import pandas as pd

train_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_metadata_supervised.csv"
test_path  = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/test_metadata_supervised.csv"

output_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/net_test_metadata.csv"

# -------------------------
# Load metadata
# -------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# -------------------------
# Extract identities
# -------------------------
def get_identity(p):
    return p.split("/")[0]

train_ids = set(train_df["path"].apply(get_identity))
test_df["identity"] = test_df["path"].apply(get_identity)

# -------------------------
# Filter test set
# -------------------------
filtered_test = test_df[~test_df["identity"].isin(train_ids)].copy()

# -------------------------
# Stats
# -------------------------
print("Original test samples:", len(test_df))
print("Filtered test samples:", len(filtered_test))
print("Removed samples:", len(test_df) - len(filtered_test))
print("Unique test identities:", test_df["identity"].nunique())
print("Remaining identities:", filtered_test["identity"].nunique())

# sanity check
overlap_ids = set(filtered_test["identity"]) & train_ids
print("Overlap identities after filtering:", len(overlap_ids))

# -------------------------
# Save
# -------------------------
filtered_test.drop(columns=["identity"]).to_csv(output_path, index=False)

print(f"\nSaved identity-disjoint test set to:\n{output_path}")