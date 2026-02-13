import os
import tensorflow as tf
import numpy as np

# =========================
# CONFIG
# =========================

train_dir = "data/processed/isimand_dataset/train"
val_dir = "data/processed/isimand_dataset/val"

image_size = (224, 224)
batch_size = 32

# =========================
# LOAD DATASETS
# =========================

print("\n[INFO] Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",  # change if needed
    shuffle=True
)

print("\n[INFO] Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False
)

# =========================
# 1️⃣ CLASS ORDER CHECK
# =========================

print("\n==============================")
print("[CHECK] Class Order")
print("==============================")

print("Train class names:", train_ds.class_names)
print("Val class names  :", val_ds.class_names)

# =========================
# 2️⃣ BATCH SHAPE CHECK
# =========================

print("\n==============================")
print("[CHECK] Batch Shape & Label Format")
print("==============================")

for images, labels in train_ds.take(1):
    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("First 5 labels:\n", labels[:5].numpy())

# =========================
# 3️⃣ PER-CLASS FILE COUNT CHECK
# =========================

print("\n==============================")
print("[CHECK] Raw File Counts (Train)")
print("==============================")

for cls in sorted(os.listdir(train_dir)):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        print(f"{cls}: {len(os.listdir(cls_path))}")

print("\n==============================")
print("[CHECK] Raw File Counts (Validation)")
print("==============================")

for cls in sorted(os.listdir(val_dir)):
    cls_path = os.path.join(val_dir, cls)
    if os.path.isdir(cls_path):
        print(f"{cls}: {len(os.listdir(cls_path))}")

# =========================
# 4️⃣ DISTRIBUTION CHECK FROM DATASET PIPELINE
# =========================

print("\n==============================")
print("[CHECK] Distribution via Dataset Iteration (Train)")
print("==============================")

class_counts = np.zeros(len(train_ds.class_names))

for _, labels in train_ds.unbatch():
    class_index = np.argmax(labels.numpy())
    class_counts[class_index] += 1

for i, name in enumerate(train_ds.class_names):
    print(f"{name}: {int(class_counts[i])}")

print("\n==============================")
print("[CHECK] Distribution via Dataset Iteration (Validation)")
print("==============================")

class_counts_val = np.zeros(len(val_ds.class_names))

for _, labels in val_ds.unbatch():
    class_index = np.argmax(labels.numpy())
    class_counts_val[class_index] += 1

for i, name in enumerate(val_ds.class_names):
    print(f"{name}: {int(class_counts_val[i])}")

print("\n[INFO] Dataset check complete.")