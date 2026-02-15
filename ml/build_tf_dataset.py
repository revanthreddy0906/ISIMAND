import tensorflow as tf
from dataset import XView2BuildingDataset
import numpy as np


def create_tf_dataset(metadata_csv, batch_size=32):
    dataset_obj = XView2BuildingDataset(metadata_csv)

    def generator():
        for i in range(len(dataset_obj)):
            sample = dataset_obj[i]
            if sample is None:
                continue
            yield sample

    output_signature = (
        tf.TensorSpec(shape=(128, 128, 6), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    tf_dataset = tf_dataset.shuffle(5000)
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset