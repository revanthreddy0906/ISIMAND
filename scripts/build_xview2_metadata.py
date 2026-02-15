import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

# --------------------------
# Paths
# --------------------------
RAW_TRAIN_DIR = "data/train"
IMAGES_DIR = os.path.join(RAW_TRAIN_DIR, "images")
LABELS_DIR = os.path.join(RAW_TRAIN_DIR, "labels")
OUTPUT_CSV = "data/xview2_metadata.csv"

rows = []

json_files = glob(os.path.join(LABELS_DIR, "*_post_disaster.json"))
print(f"Found {len(json_files)} post-disaster JSON files")

# --------------------------
# Label Mapping Function
# --------------------------
def map_damage_label(label):
    if label == "no-damage":
        return 0
    elif label in ["minor-damage", "major-damage"]:
        return 1
    elif label == "destroyed":
        return 2
    else:
        return None

# --------------------------
# Main Loop
# --------------------------
for json_path in tqdm(json_files):

    tile_id = os.path.basename(json_path).replace("_post_disaster.json", "")

    pre_image_path = os.path.join(IMAGES_DIR, f"{tile_id}_pre_disaster.png")
    post_image_path = os.path.join(IMAGES_DIR, f"{tile_id}_post_disaster.png")

    if not os.path.exists(pre_image_path) or not os.path.exists(post_image_path):
        continue

    with open(json_path) as f:
        data = json.load(f)

    for building_id, feature in enumerate(data["features"]["xy"]):

        original_label = feature["properties"]["subtype"]
        mapped_label = map_damage_label(original_label)

        # Skip un-classified
        if mapped_label is None:
            continue

        rows.append({
            "tile_id": tile_id,
            "json_path": json_path,
            "pre_image_path": pre_image_path,
            "post_image_path": post_image_path,
            "building_id": building_id,
            "damage_class": mapped_label
        })

# --------------------------
# Save CSV
# --------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Metadata file created.")
print("Total buildings:", len(df))
print("\nClass distribution:")
print(df["damage_class"].value_counts())