import tensorflow as tf


def sparse_categorical_focal_loss(gamma=2.0):

    def loss_fn(y_true, y_pred):
        # Ensure integer labels
        y_true = tf.cast(y_true, tf.int32)

        # Convert to one-hot
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        # Clip predictions to prevent log(0)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute standard cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        # Compute focal scaling factor
        focal_factor = tf.pow(1 - y_pred, gamma)

        # Apply focal factor
        loss = focal_factor * cross_entropy

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss

    return loss_fn