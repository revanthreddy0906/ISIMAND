import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import XView2BuildingDataset
from losses import WeightedFocalLoss


# =========================
# CONFIG
# =========================
METADATA_PATH = "data/xview2_metadata.csv"
MODEL_SAVE_PATH = "ml/models/efficientnet_stage1.keras"

PATCH_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4


# =========================
# CREATE TF.DATA DATASET
# =========================

def create_tf_dataset(metadata_csv, batch_size):

    dataset_obj = XView2BuildingDataset(metadata_csv, patch_size=PATCH_SIZE)

    def generator():
        for i in range(len(dataset_obj)):
            sample = dataset_obj[i]
            if sample is None:
                continue
            yield sample

    output_signature = (
        tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    tf_dataset = tf_dataset.shuffle(10000)
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


print("Building dataset...")
train_ds = create_tf_dataset(METADATA_PATH, BATCH_SIZE)


# =========================
# BUILD MODEL
# =========================

def build_model():

    inputs = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 3))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    base_model.trainable = False  # Stage-1 freeze

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


model = build_model()


# =========================
# LOSS FUNCTION
# =========================

loss_fn = WeightedFocalLoss(
    alpha=[0.15, 0.35, 0.5],  # no_damage, severe, destroyed
    gamma=2.0
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=["accuracy"]
)


# =========================
# CALLBACKS
# =========================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    save_best_only=True,
    monitor="loss"
)


# =========================
# TRAIN
# =========================

print("Starting Stage-1 training...")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

print("Stage-1 training complete.")
print(f"Model saved to {MODEL_SAVE_PATH}")