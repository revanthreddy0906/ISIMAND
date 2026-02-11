from cnn_model import build_cnn_model
import tensorflow as tf
import os


batch_size = 32
epochs = 20
image_size = (224,224)
learning_rate = 0.001

train_dir = "data/processed/isimand_dataset/train"
val_dir = "data/processed/isimand_dataset/val"

model_save_path = "ml/models/cnn_model.h5"


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

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x),y))

class_weights = {
    0: 0.46,   # no_damage
    1: 9.82,   # moderate_damage
    2: 1.42    # severe_damage
}

model = build_cnn_model()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
)

os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
model.save(model_save_path)

print(f"[INFO] Training completed. Model saved to {model_save_path}")

