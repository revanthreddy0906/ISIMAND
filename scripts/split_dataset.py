import os
import random
import shutil

base_dir = "data/processed"
output_dir = "data/processed/isimand_dataset"

classes = ["no_damage","moderate_damage","severe_damage"]

train_ratio = 0.7
val_ratio =0.15
test_ratio = 0.15

for split in ["train","val","test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir,split,cls),exist_ok=True)

print("[INFO] Output split directories ready")
for cls in classes:
    class_dir = os.path.join(base_dir,cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    print(f"\nClass: {cls}")
    print(f"Train: {len(train_files)}")
    print(f"Val  : {len(val_files)}")
    print(f"Test : {len(test_files)}")

    for split_name, split_files in [
        ("train",train_files),
        ("val",val_files),
        ("test",test_files)
    ]:
        for file in split_files:
            src = os.path.join(class_dir,file)
            dst = os.path.join(output_dir,split_name,cls,file)
            shutil.copy2(src,dst)
        print(f"Copied {len(split_files)} images to {split_name}/{cls}")

print("\n[INFO] Dataset split completed")



