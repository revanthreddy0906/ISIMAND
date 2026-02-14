import tensorflow as tf
import os
from efficientnet_model import build_efficientnet_model

batch_size = 32
epochs = 20
image_size = (224,224)
learning_rate = 5e-4

train_dir = "data/processed/isimand_dataset/train"
val_dir = "data/processed/isimand_dataset/val"

model_save_path = "ml/models/efficientnet_stage1.keras"

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

class_names = train_ds.class_names
print(class_names)

class_weights = {
    0: 12.4,
    1: 0.42,
    2: 1.8
}

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

model, base_model = build_efficientnet_model()

model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

print("Starting EfficientNet Training")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping],
    class_weight=class_weights
)

os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
model.save(model_save_path)

print(f"[INFO] Training completed. Model saved to {model_save_path}")