SRC="/egr/research-sprintai/baliahsa/projects/AVH-Align/Features/AV1M-Trimmed/test"

DST="/egr/research-sprintai/baliahsa/projects/AVH-Align/Features/AV1M-Base-Trimmed/test"

find "$SRC" -type f -name "*labels.npz" | while read -r file; do

    rel_path="${file#$SRC/}"

    dst_file="$DST/$rel_path"

    mkdir -p "$(dirname "$dst_file")"

    cp -f "$file" "$dst_file"

done