import tensorflow as tf
import math

class ResizeAndNormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size=(300, 300), **kwargs):
        super(ResizeAndNormalizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        images, keypoints = inputs
        original_size = tf.shape(images)[1:3]
        images = tf.image.resize(images, self.target_size)
        
        # Normalize keypoints
        scale = tf.cast(self.target_size, tf.float32) / tf.cast(original_size, tf.float32)
        keypoints_xy = keypoints[:, :, :2] * scale
        keypoints = tf.concat([keypoints_xy, keypoints[:, :, 2:]], axis=-1)
        
        return images, keypoints
    

class AugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        images, keypoints = inputs
        images, keypoints = self.random_flip(images, keypoints)
        images, keypoints = self.random_rotation(images, keypoints)
        images, keypoints = self.random_crop(images, keypoints)
        images, keypoints = self.random_brightness(images, keypoints)
        
        return images, keypoints

    def random_flip(self, image, keypoints):
        flip_cond = tf.random.uniform([]) > 0.5
        image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(image), lambda: image)
        keypoints = tf.cond(flip_cond, lambda: tf.concat([1.0 - keypoints[:, :, :1], keypoints[:, :, 1:]], axis=-1), lambda: keypoints)
        return image, keypoints

    def random_rotation(self, image, keypoints, max_angle=15):
        angle = tf.random.uniform([], -max_angle, max_angle)
        image = tf.image.rotate(image, angle * tf.constant(math.pi / 180, dtype=tf.float32))

        center = tf.constant([0.5, 0.5], dtype=tf.float32)
        keypoints_xy = keypoints[:, :, :2] - center
        angle_rad = angle * tf.constant(math.pi / 180, dtype=tf.float32)
        rotation_matrix = tf.stack([
            [tf.cos(angle_rad), -tf.sin(angle_rad)],
            [tf.sin(angle_rad), tf.cos(angle_rad)]
        ])
        keypoints_xy = tf.matmul(keypoints_xy, rotation_matrix) + center
        keypoints = tf.concat([keypoints_xy, keypoints[:, :, 2:]], axis=-1)

        return image, keypoints

    def random_crop(self, image, keypoints):
        crop_size = tf.random.uniform([], 0.8, 1.0)
        h, w = tf.shape(image)[1], tf.shape(image)[2]
        crop_h, crop_w = tf.cast(h * crop_size, tf.int32), tf.cast(w * crop_size, tf.int32)
        image = tf.image.random_crop(image, size=[tf.shape(image)[0], crop_h, crop_w, 3])

        scale_h, scale_w = tf.cast(crop_h, tf.float32) / tf.cast(h, tf.float32), tf.cast(crop_w, tf.float32) / tf.cast(w, tf.float32)
        keypoints_xy = keypoints[:, :, :2] * tf.constant([scale_w, scale_h], dtype=tf.float32)
        keypoints = tf.concat([keypoints_xy, keypoints[:, :, 2:]], axis=-1)

        return image, keypoints

    def random_brightness(self, image, keypoints):
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image, keypoints