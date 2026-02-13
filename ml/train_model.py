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


moderate_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(train_dir, "moderate_damage"),
    image_size=image_size,
    batch_size=None,
    labels=None,
    shuffle=True
)
moderate_ds = moderate_ds.map(lambda x: (x, tf.constant([1.0, 0.0, 0.0])))

# 2. No Damage (Index 1 -> [0, 1, 0])
no_damage_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(train_dir, "no_damage"),
    image_size=image_size,
    batch_size=None,
    labels=None,
    shuffle=True
)
no_damage_ds = no_damage_ds.map(lambda x: (x, tf.constant([0.0, 1.0, 0.0])))

# 3. Severe Damage (Index 2 -> [0, 0, 1])
severe_damage_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(train_dir, "severe_damage"),
    image_size=image_size,
    batch_size=None,
    labels=None,
    shuffle=True
)
severe_damage_ds = severe_damage_ds.map(lambda x: (x, tf.constant([0.0, 0.0, 1.0])))

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

moderate_ds = moderate_ds.map(lambda x,y: (normalization_layer(x),y))
no_damage_ds = no_damage_ds.map(lambda x,y: (normalization_layer(x),y))
severe_damage_ds = severe_damage_ds.map(lambda x,y: (normalization_layer(x),y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x),y))

# Oversample the minority classes
moderate_ds = moderate_ds.repeat()
no_damage_ds = no_damage_ds.repeat()
severe_damage_ds = severe_damage_ds.repeat()

balanced_train_ds = tf.data.Dataset.sample_from_datasets(
    [moderate_ds, no_damage_ds, severe_damage_ds],
    weights=[0.5, 0.25, 0.25]  # moderate boosted
)
balanced_train_ds = balanced_train_ds.batch(batch_size)
balanced_train_ds = balanced_train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])


model = build_cnn_model(data_augmentation)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)
steps_per_epoch = 300

history = model.fit(
    balanced_train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stopping]
)

os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
model.save(model_save_path)

print(f"[INFO] Training completed. Model saved to {model_save_path}")

