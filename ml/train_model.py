from cnn_model import build_cnn_model
import tensorflow as tf
import os



batch_size = 32
epochs = 20
image_size = (224,224)
learning_rate = 5e-5

train_dir = "data/processed/isimand_dataset/train"
val_dir = "data/processed/isimand_dataset/val"

model_save_path = "ml/models/cnn_model.h5"


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)
class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x),y))

class_weights = {
    0: 9.82,   # moderate_damage
    1: 0.46,   # no_damage
    2: 1.42    # severe_damage
}

print(class_names)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

model = build_cnn_model(data_augmentation)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
model.save(model_save_path)

print(f"[INFO] Training completed. Model saved to {model_save_path}")

