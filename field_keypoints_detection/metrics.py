
import tensorflow as tf

class VisibleKeypointMSE(tf.keras.metrics.Metric):
    def __init__(self, name='visible_keypoint_mse', **kwargs):
        super(VisibleKeypointMSE, self).__init__(name=name, **kwargs)
        self.total_mse = self.add_weight(name='total_mse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        visibility_mask = tf.cast(y_true[:, :, 2] != 0, dtype=tf.float32)
        y_true_points = y_true[:, :, :2]
        y_pred_points = y_pred[:, :, :2]

        mse = tf.reduce_sum(tf.square(y_true_points - y_pred_points), axis=-1)
        mse = mse * visibility_mask

        self.total_mse.assign_add(tf.reduce_sum(mse))
        self.count.assign_add(tf.reduce_sum(visibility_mask))

    def result(self):
        return self.total_mse / self.count

    def reset_states(self):
        self.total_mse.assign(0.0)
        self.count.assign(0.0)

class VisibilityAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='visibility_accuracy', **kwargs):
        super(VisibilityAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_visibility = tf.cast(y_true[:, :, 2], dtype=tf.int32)
        y_pred_visibility = tf.cast(tf.round(y_pred[:, :, 2]), dtype=tf.int32)

        correct = tf.reduce_sum(tf.cast(tf.equal(y_true_visibility, y_pred_visibility), dtype=tf.float32))
        total = tf.size(y_true_visibility, out_type=tf.float32)

        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)