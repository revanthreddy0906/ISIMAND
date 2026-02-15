import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
METADATA_PATH = "data/xview2_metadata.csv"
OUTPUT_ROOT = "data/buildings_diff"
PATCH_SIZE = 128

CLASS_MAP = {
    0: "no_damage",
    1: "severe_damage",
    2: "destroyed"
}

# =========================
# SETUP OUTPUT FOLDERS
# =========================
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for class_name in CLASS_MAP.values():
    os.makedirs(os.path.join(OUTPUT_ROOT, class_name), exist_ok=True)


# =========================
# HELPER FUNCTION
# =========================
def crop_polygon(image, polygon):
    """
    Crop building polygon region from image.
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)

    cv2.fillPoly(mask, [pts], 255)

    x, y, w, h = cv2.boundingRect(pts)

    cropped = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    cropped[cropped_mask == 0] = 0

    return cropped


# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(METADATA_PATH)

print("Building difference dataset...")
print("Total buildings:", len(df))

error_count = 0

# =========================
# MAIN LOOP
# =========================
for row in tqdm(df.itertuples(index=False), total=len(df)):

    try:
        tile_id = row.tile_id
        json_path = row.json_path
        building_id = row.building_id
        label = row.damage_class

        class_name = CLASS_MAP[label]

        pre_img = cv2.imread(row.pre_image_path)
        post_img = cv2.imread(row.post_image_path)

        if pre_img is None or post_img is None:
            error_count += 1
            continue

        with open(json_path) as f:
            data = json.load(f)

        building = data["features"]["xy"][building_id]
        polygon = building["geometry"]["coordinates"][0]

        pre_crop = crop_polygon(pre_img, polygon)
        post_crop = crop_polygon(post_img, polygon)

        if pre_crop.size == 0 or post_crop.size == 0:
            error_count += 1
            continue

        # Resize
        pre_crop = cv2.resize(pre_crop, (PATCH_SIZE, PATCH_SIZE))
        post_crop = cv2.resize(post_crop, (PATCH_SIZE, PATCH_SIZE))

        # Normalize
        pre_crop = pre_crop.astype(np.float32) / 255.0
        post_crop = post_crop.astype(np.float32) / 255.0

        # Difference
        diff = post_crop - pre_crop

        # Shift range from [-1,1] → [0,1]
        diff = (diff + 1.0) / 2.0
        diff = np.clip(diff, 0, 1)

        # Convert to uint8
        diff = (diff * 255).astype(np.uint8)

        filename = f"{tile_id}_{building_id}.png"
        save_path = os.path.join(OUTPUT_ROOT, class_name, filename)

        cv2.imwrite(save_path, diff)

    except Exception:
        error_count += 1
        continue


print("\nDataset build complete.")
print("Errors skipped:", error_count)