import json
import os
import pandas as pd

# Input metadata path
metadata_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/LAV-DF/LAV-DF/metadata.json"

# Output directory
output_dir = "/egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/LAV-DF/LavDF_metadata"
os.makedirs(output_dir, exist_ok=True)

# Load original metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

train_rows = []
test_rows = []

for item in metadata:

    # Path
    path = item["file"]

    # Label
    # 0 -> real
    # 1 -> fake
    label = 0 if item["n_fakes"] == 0 else 1

    # Number of frames
    num_frames = item["video_frames"]

    row = {
        "path": path,
        "label": label,
        "num_frames": num_frames
    }

    # Split
    if item["split"] == "train":
        train_rows.append(row)

    elif item["split"] == "test":
        test_rows.append(row)

# Create dataframes
train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# Save CSVs
train_csv = os.path.join(output_dir, "train_metadata.csv")
test_csv = os.path.join(output_dir, "test_metadata.csv")

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Saved train metadata: {train_csv}")
print(f"Saved test metadata: {test_csv}")

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")