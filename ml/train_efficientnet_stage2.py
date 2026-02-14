import tensorflow as tf
import os
from losses import sparse_categorical_focal_loss
from efficientnet_model import build_efficientnet_model

# -----------------------
# Config
# -----------------------
batch_size = 32
epochs = 12
image_size = (224, 224)
learning_rate = 1e-5

train_dir = "data/processed/isimand_dataset/train"
val_dir = "data/processed/isimand_dataset/val"

stage1_model_path = "ml/models/efficientnet_stage1.keras"
stage2_model_path = "ml/models/efficientnet_stage2.keras"

# -----------------------
# Load Dataset (int labels)
# -----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------
# Load Stage-1 Model
# -----------------------
model = tf.keras.models.load_model(stage1_model_path)

# Access base model (EfficientNet)
base_model = model.layers[1]  # backbone is second layer

# -----------------------
# Unfreeze last 30 layers
# -----------------------
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

print("Trainable layers after unfreezing:")
for layer in base_model.layers[-30:]:
    print(layer.name, layer.trainable)

# -----------------------
# Compile with small LR
# -----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=sparse_categorical_focal_loss(gamma=2.0),
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# -----------------------
# Fine-Tune
# -----------------------
print("Starting Stage-2 Fine-Tuning...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)

# -----------------------
# Save
# -----------------------
os.makedirs(os.path.dirname(stage2_model_path), exist_ok=True)
model.save(stage2_model_path)

print(f"[INFO] Stage-2 model saved to {stage2_model_path}")