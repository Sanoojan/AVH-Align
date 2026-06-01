import os
import json
import random
import pandas as pd
random.seed(42)
# Paths
csv_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/test_metadata_cleaned.csv"
json_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val_metadata.json"
output_path = "/egr/research-sprintai/baliahsa/projects/AVH-Align/av1m_metadata/test_metadata_supervised.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    metadata_json = json.loads(f.read(), strict=False)

# Build lookup dictionary from JSON
# Key: file path (relative path)
# Value: video_frames
json_lookup = {}
for item in metadata_json:
    json_lookup[item["file"]] = item["video_frames"]

# Fake variants to randomly select from
fake_variants = [
    "real_video_fake_audio.mp4",
    "fake_video_real_audio.mp4",
    "fake_video_fake_audio.mp4"
]

new_rows = []

for _, row in df.iterrows():
    original_path = row["path"]  # idXXXX/.../real.mp4
    video_dir = os.path.dirname(original_path)
    label = row["label"]
    json_key = os.path.join("vox_celeb_2", original_path)
    new_rows.append({
        "path": original_path,
        "label": label,
        "num_frames": json_lookup[json_key]
    })


# Convert new rows to dataframe
df_final= pd.DataFrame(new_rows)

# Concatenate real + fake
# df_final = pd.concat([df, df_fake], ignore_index=True)

# Save
df_final.to_csv(output_path, index=False)

print(f"Saved extended metadata to: {output_path}")
print(f"Original samples: {len(df)}")
print(f"Total samples: {len(df_final)}")