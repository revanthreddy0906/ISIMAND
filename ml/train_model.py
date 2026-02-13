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
    label_mode="categorical"
)
class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x),y))

def make_class_dataset(class_index):
    return train_ds.filter(
        lambda x,y: tf.argmax(y) == class_index
    )

moderate_ds = make_class_dataset(0)
no_damage_ds = make_class_dataset(1)
severe_damage_ds = make_class_dataset(2)

#Oversample the minority classes
moderate_ds = moderate_ds.repeat()
severe_damage_ds = severe_damage_ds.repeat()
no_damage_ds = no_damage_ds.repeat()

balanced_train_ds = tf.data.Dataset.sample_from_datasets(
    [moderate_ds, no_damage_ds, severe_damage_ds],
    weights=[0.4, 0.3, 0.3]  # moderate boosted
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
    patience=5,
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

