import os
import pandas as pd

# Paths
metadata_path = "av1m_metadata/test_metadata.csv"
main_folder = "/egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/Audio_Video_Preprocess/AV_Deepfake1M/test"

# Output files
clean_metadata_path = "av1m_metadata/test_metadata_cleaned.csv"
removed_metadata_path = "av1m_metadata/test_metadata_removed.csv"

# Load metadata
df = pd.read_csv(metadata_path)

valid_rows = []
removed_rows = []

for _, row in df.iterrows():
    relative_path = row["path"]
    
    # replace .mp4 → _roi.mp4
    modified_path = relative_path.replace(".mp4", "_roi.mp4")
    full_path = os.path.join(main_folder, modified_path)

    if os.path.exists(full_path):
        valid_rows.append(row)
    else:
        removed_rows.append(row)

# Save cleaned metadata
pd.DataFrame(valid_rows).to_csv(clean_metadata_path, index=False)

# Save removed metadata
pd.DataFrame(removed_rows).to_csv(removed_metadata_path, index=False)

print(f"Original entries: {len(df)}")
print(f"Valid entries: {len(valid_rows)}")
print(f"Removed entries: {len(removed_rows)}")