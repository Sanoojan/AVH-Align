import json
import os
from collections import Counter

train_path = "/egr/research-sprintai/shared/Datasets-Vision/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/train_metadata.json"
val_path   = "/egr/research-sprintai/shared/Datasets-Vision/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val_metadata.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_speaker_id(path):
    """
    Extract speaker id (e.g., id01358)
    from:
    vox_celeb_2/id01358/_1nATum8x78/00030/real.mp4
    """
    parts = path.split("/")
    for p in parts:
        if p.startswith("id"):
            return p
    return None


# Load metadata
train_data = load_json(train_path)
val_data   = load_json(val_path)

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")


# -----------------------------
# 1. File-level leakage
# -----------------------------
train_files = set(item["file"] for item in train_data)
val_files   = set(item["file"] for item in val_data)

file_overlap = train_files.intersection(val_files)

print("\n=== File-level Leakage ===")
print(f"Overlapping files: {len(file_overlap)}")


# -----------------------------
# 2. Original-video leakage
# -----------------------------
train_originals = set(item["original"] for item in train_data)
val_originals   = set(item["original"] for item in val_data)

original_overlap = train_originals.intersection(val_originals)

print("\n=== Original Video Leakage ===")
print(f"Overlapping originals: {len(original_overlap)}")


# -----------------------------
# 3. Speaker-level leakage
# -----------------------------
train_speakers = set(extract_speaker_id(item["file"]) for item in train_data)
val_speakers   = set(extract_speaker_id(item["file"]) for item in val_data)

print("\nUnique speakers in train set:", len(train_speakers))
print("Unique speakers in val set:", len(val_speakers))

speaker_overlap = train_speakers.intersection(val_speakers)

print("\n=== Speaker-level Leakage ===")
print(f"Overlapping speakers: {len(speaker_overlap)}")

if len(speaker_overlap) > 0:
    print("Example overlapping speakers:", list(speaker_overlap)[:10])


# -----------------------------
# 4. Distribution sanity check
# -----------------------------
train_modify = Counter(item["modify_type"] for item in train_data)
val_modify   = Counter(item["modify_type"] for item in val_data)

print("\n=== Modify Type Distribution ===")
print("Train:", train_modify)
print("Val:  ", val_modify)


# -----------------------------
# 5. Split field consistency
# -----------------------------
wrong_train_split = [x for x in train_data if x["split"] != "train"]
wrong_val_split   = [x for x in val_data if x["split"] != "val"]

print("\n=== Split Field Check ===")
print(f"Wrong entries in train file: {len(wrong_train_split)}")
print(f"Wrong entries in val file:   {len(wrong_val_split)}")


print("\nDone checking split integrity.")