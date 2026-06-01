import pandas as pd
from collections import Counter

train_csv = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_split.csv"
# test_csv  = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/test_metadata_cleaned.csv"
test_csv   = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/val_split.csv"  # IGNORE
# -------------------------------------------------
# Load
# -------------------------------------------------
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)
# val_df   = pd.read_csv(val_csv)

print("Train samples:", len(train_df))
print("Test samples :", len(test_df))
print()

# -------------------------------------------------
# 1️⃣ Exact path overlap
# -------------------------------------------------
train_paths = set(train_df["path"])
test_paths  = set(test_df["path"])

exact_overlap = train_paths.intersection(test_paths)

print("Exact clip overlap:", len(exact_overlap))

# -------------------------------------------------
# 2️⃣ Identity overlap (idXXXXX)
# -------------------------------------------------
def extract_identity(path):
    return path.split("/")[0]

train_ids = set(train_df["path"].apply(extract_identity))
test_ids  = set(test_df["path"].apply(extract_identity))

print("Unique identities in train set:", len(train_ids))
print("Unique identities in test set:", len(test_ids))

identity_overlap = train_ids.intersection(test_ids)

print("Identity overlap:", len(identity_overlap))
print("Example overlapping identities:", list(identity_overlap)[:10])
print()

# -------------------------------------------------
# 3️⃣ YouTube ID overlap (second folder)
# -------------------------------------------------
def extract_youtube_id(path):
    return path.split("/")[1]

train_yt = set(train_df["path"].apply(extract_youtube_id))
test_yt  = set(test_df["path"].apply(extract_youtube_id))

youtube_overlap = train_yt.intersection(test_yt)

print("YouTube ID overlap:", len(youtube_overlap))
print("Example overlapping YT IDs:", list(youtube_overlap)[:10])
print()

# -------------------------------------------------
# 4️⃣ Identity + YouTube pair overlap
# -------------------------------------------------
def extract_id_yt(path):
    parts = path.split("/")
    return parts[0] + "/" + parts[1]

train_pairs = set(train_df["path"].apply(extract_id_yt))
test_pairs  = set(test_df["path"].apply(extract_id_yt))

pair_overlap = train_pairs.intersection(test_pairs)

print("Identity+YouTube pair overlap:", len(pair_overlap))
print("Example overlapping pairs:", list(pair_overlap)[:10])
print()

# -------------------------------------------------
# Summary Verdict
# -------------------------------------------------
print("-------------------------------------------------")
if len(identity_overlap) > 0:
    print("⚠ WARNING: Same identities appear in train and test!")
else:
    print("✅ No identity leakage.")

if len(youtube_overlap) > 0:
    print("⚠ WARNING: Same YouTube videos appear in train and test!")
else:
    print("✅ No YouTube video leakage.")

if len(pair_overlap) > 0:
    print("⚠ WARNING: Same identity+YouTube pairs appear in both splits!")
else:
    print("✅ No identity+YouTube leakage.")

if len(exact_overlap) > 0:
    print("🚨 CRITICAL: Exact clip duplicates found!")
else:
    print("✅ No exact clip duplicates.")