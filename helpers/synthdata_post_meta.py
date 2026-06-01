import os
import csv

REAL_METADATA = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_metadata.csv"
FEATURE_ROOT = "/egr/research-sprintai/baliahsa/projects/AVH-Align/Features/AV1M/train"
OUTPUT_METADATA = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/train_metadata_with_synth_fake_segments.csv"

VALID_PREFIXES = [
    "real_synthfake_audio",
    "real_synthfake_video",
    "real_synthfake_both"
]

# -------------------------------------------------
# 1. Load real metadata
# -------------------------------------------------
real_dict = {}

with open(REAL_METADATA, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        real_dict[row["path"]] = {
            "label": int(row["label"]),
            "num_frames": int(row["num_frames"])
        }

print(f"Loaded {len(real_dict)} real entries")

# -------------------------------------------------
# 2. Scan synthetic features
# -------------------------------------------------
synthetic_entries = []

for root, _, files in os.walk(FEATURE_ROOT):
    for file in files:

        # skip non-npz + label files
        if not file.endswith(".npz"):
            continue
        if "labels" in file:
            continue

        # strict filtering
        if not any(file.startswith(prefix) for prefix in VALID_PREFIXES):
            continue

        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, FEATURE_ROOT)

        fake_video_path = rel_path.replace(".npz", ".mp4")

        # map to real video
        real_video_path = fake_video_path
        for prefix in VALID_PREFIXES:
            real_video_path = real_video_path.replace(prefix, "real")

        if real_video_path not in real_dict:
            print(f"[WARNING] Missing real match: {fake_video_path}")
            continue

        synthetic_entries.append({
            "path": fake_video_path,
            "label": 1,
            "num_frames": real_dict[real_video_path]["num_frames"]
        })

print(f"Found {len(synthetic_entries)} synthetic entries")

# -------------------------------------------------
# 3. Write final metadata
# -------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_METADATA), exist_ok=True)

seen = set()

with open(OUTPUT_METADATA, "w", newline="") as f:
    fieldnames = ["path", "label", "num_frames"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # --- add synthetic first ---
    for entry in synthetic_entries:
        if entry["path"] not in seen:
            writer.writerow(entry)
            seen.add(entry["path"])

    # --- add ALL original metadata ---
    with open(REAL_METADATA, "r") as f_real:
        reader = csv.DictReader(f_real)
        for row in reader:
            if row["path"] not in seen:
                writer.writerow({
                    "path": row["path"],
                    "label": int(row["label"]),
                    "num_frames": int(row["num_frames"])
                })
                seen.add(row["path"])

print(f"Final metadata saved to: {OUTPUT_METADATA}")