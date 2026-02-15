import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------

RAW_TRAIN_DIR = "data/train"
IMAGES_DIR = os.path.join(RAW_TRAIN_DIR, "images")
LABELS_DIR = os.path.join(RAW_TRAIN_DIR, "labels")

OUTPUT_DIR = "data/processed_buildings"
IMAGE_SIZE = (224, 224)

# Damage mapping (3-class)
DAMAGE_MAP = {
    "no-damage": "no_damage",
    "minor-damage": "moderate_damage",
    "major-damage": "moderate_damage",
    "destroyed": "severe_damage"
}

# ------------------------
# Create output folders
# ------------------------

for cls in set(DAMAGE_MAP.values()):
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)


# ------------------------
# Helper: get bounding box from polygon
# ------------------------

def polygon_to_bbox(polygon_wkt):
    """
    Convert WKT POLYGON string to bounding box.
    """
    coords = polygon_wkt.replace("POLYGON ((", "").replace("))", "")
    points = coords.split(",")

    xs = []
    ys = []

    for point in points:
        x, y = point.strip().split(" ")
        xs.append(float(x))
        ys.append(float(y))

    xmin = int(min(xs))
    xmax = int(max(xs))
    ymin = int(min(ys))
    ymax = int(max(ys))

    return xmin, ymin, xmax, ymax


# ------------------------
# Main Processing Loop
# ------------------------

json_files = [f for f in os.listdir(LABELS_DIR) if f.endswith("_post_disaster.json")]

print(f"Found {len(json_files)} post-disaster JSON files")

sample_counter = 0

for json_file in tqdm(json_files):

    json_path = os.path.join(LABELS_DIR, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Get matching image names
    base_name = json_file.replace("_post_disaster.json", "")

    pre_image_path = os.path.join(IMAGES_DIR, base_name + "_pre_disaster.png")
    post_image_path = os.path.join(IMAGES_DIR, base_name + "_post_disaster.png")

    if not os.path.exists(pre_image_path) or not os.path.exists(post_image_path):
        continue

    pre_img = cv2.imread(pre_image_path)
    post_img = cv2.imread(post_image_path)

    if pre_img is None or post_img is None:
        continue

    features = data.get("features", {}).get("xy", [])

    for idx, feature in enumerate(features):

        subtype = feature.get("properties", {}).get("subtype", "no-damage")

        if subtype not in DAMAGE_MAP:
            continue

        class_name = DAMAGE_MAP[subtype]

        polygon_wkt = feature.get("wkt", None)

        if polygon_wkt is None:
            continue

        xmin, ymin, xmax, ymax = polygon_to_bbox(polygon_wkt)

        # Safety bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(pre_img.shape[1], xmax)
        ymax = min(pre_img.shape[0], ymax)

        if xmax - xmin < 5 or ymax - ymin < 5:
            continue

        # Crop
        pre_crop = pre_img[ymin:ymax, xmin:xmax]
        post_crop = post_img[ymin:ymax, xmin:xmax]

        # Resize
        pre_crop = cv2.resize(pre_crop, IMAGE_SIZE)
        post_crop = cv2.resize(post_crop, IMAGE_SIZE)

        # Stack into 6-channel
        stacked = np.concatenate([pre_crop, post_crop], axis=2)

        # Save
        save_name = f"{base_name}_building_{idx}.npy"
        save_path = os.path.join(OUTPUT_DIR, class_name, save_name)

        np.save(save_path, stacked)

        sample_counter += 1

print(f"\nDataset creation complete.")
print(f"Total building samples created: {sample_counter}")