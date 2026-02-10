import os
import json
from PIL import Image

raw_images_dir ="data/unprocessed/images"
raw_labels_dir ="data/unprocessed/labels"

output_base_dir = "data/processed"
no_damage_dir = os.path.join(output_base_dir, "no_damage")
moderate_damage_dir = os.path.join(output_base_dir, "moderate_damage")
severe_damage_dir = os.path.join(output_base_dir, "severe_damage")

Image_Size = (224, 224)

os.makedirs(no_damage_dir, exist_ok=True)
os.makedirs(moderate_damage_dir, exist_ok=True)
os.makedirs(severe_damage_dir, exist_ok=True)

def get_image_damage_label(json_path):
    severity_map = {
        "no-damage": 0,
        "un-classified": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3
    }
    try:
        with open(json_path,"r") as f:
            data=json.load(f)
    except Exception as e:
        print(f"[WARN!!!] Error Loading JSON {json_path}:{e}")
        return "no_damage"

    features = data.get("features",{}).get("xy",[])
    highest_severity = 0

    for feature in features:
        subtype = feature.get("properties",{}).get("subtype","no_damage")
        severity = severity_map.get(subtype,0)
        highest_severity = max(highest_severity,severity)

    if highest_severity == 0:
        return "no_damage"
    elif highest_severity == 1:
        return "moderate_damage"
    else:
        return "severe_damage"

image_files = [
    f for f in os.listdir(raw_images_dir)
    if f.lower().endswith((".png",".jpeg",".jpg"))
]

print(f"Found {len(image_files)} images")

count_no = 0
count_mod = 0
count_sev = 0

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(raw_images_dir, image_file)
    json_file = os.path.splitext(image_file)[0] + ".json"
    json_path = os.path.join(raw_labels_dir, json_file)

    if not os.path.exists(json_path):
        print(f"[WARN!!!] JSON file not found for {image_path}")
        continue
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(Image_Size)
    except Exception as e:
        print(f"[WARN!!!] Error loading image {image_path}:{e}")
        continue
    label = get_image_damage_label(json_path)
    if label == "no_damage":
        count_no += 1
        save_dir = no_damage_dir
    elif label == "moderate_damage":
        count_mod += 1
        save_dir = moderate_damage_dir
    else:
        count_sev += 1
        save_dir = severe_damage_dir

    save_path = os.path.join(save_dir, image_file)
    image.save(save_path)
    print(f"Processed {idx+1}/{len(image_files)} images")

print("Dataset prepared successfully")
print(f"Processed {len(image_files)} images")
print(f"No Damage: {count_no}")
print(f"Moderate Damage: {count_mod}")
print(f"Severe Damage: {count_sev}")
