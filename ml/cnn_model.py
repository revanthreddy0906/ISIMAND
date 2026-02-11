import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Block 2
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Block 3
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Block 4
    tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Transition
    tf.keras.layers.GlobalAveragePooling2D(),

    # Classifier
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.summary()