import tensorflow as tf

class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Convert integer labels to one-hot
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        # Clip predictions
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Focal term
        focal_weight = tf.pow(1 - y_pred, self.gamma)

        # Alpha weight
        alpha_weight = y_true * self.alpha

        loss = alpha_weight * focal_weight * ce

        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))