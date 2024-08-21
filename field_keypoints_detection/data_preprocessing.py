import tensorflow as tf

class ResizeAndNormalize:
    def __init__(self, target_size=(300, 300)):
        self.target_size = target_size

    def apply(self, images, keypoints):
        # Handle the shape of the images
        original_size = tf.cast(tf.shape(images)[1:3], tf.float32)
        images = tf.image.resize(images, self.target_size)

        # Compute the scale factor for resizing keypoints
        scale = tf.cast([self.target_size[1], self.target_size[0]], tf.float32) / (original_size[1], original_size[0])

        # Resize keypoints by scaling y and x coordinates and normalize them
        keypoints_xy = (keypoints[:, :, :2] * scale) / tf.cast(self.target_size, tf.float32)
        keypoints = tf.concat([keypoints_xy, keypoints[:, :, 2:]], axis=-1)

        return images, keypoints


class Augmentation:
    def __init__(self, rescale_size=(300,300), crop_size=(0.8, 1.0), max_delta=0.2):
        self.rescale_size = rescale_size
        self.crop_size = crop_size
        self.max_delta = max_delta

    def apply(self, images, keypoints):
        # Map over each image-keypoints pair
        # Define output signature for tf.map_fn
        output_signature = (
            tf.TensorSpec(shape=tf.TensorShape([None, None, 3]), dtype=tf.float32),
            tf.TensorSpec(shape=tf.TensorShape([None, 3]), dtype=tf.float32)
        )
        
        # Map over each image-keypoints pair with output_signature
        images, keypoints = tf.map_fn(
            lambda x: self._apply_single(*x),
            (images, keypoints),
            fn_output_signature=output_signature
        )
        return images, keypoints

    def _apply_single(self, image, keypoints):
        h = tf.cast(self.rescale_size[0], tf.float32)
        w = tf.cast(self.rescale_size[1], tf.float32)

        if self.crop_size != (1.0, 1.0):
            image, keypoints = self.random_crop(image, keypoints)

        if self.max_delta != 0:
            image, keypoints = self.random_brightness(image, keypoints)
          
        image,keypoints = self.rescale_image(image, keypoints, h, w)  # Rescale after cropping and translation
        return image, keypoints

    def random_crop(self, image, keypoints):
        crop_size = tf.random.uniform([], self.crop_size[0], self.crop_size[1])
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)

        crop_h = tf.cast(h * crop_size, tf.int32)
        crop_w = tf.cast(w * crop_size, tf.int32)

        crop_offset_y = tf.random.uniform([], 0, tf.cast(h, tf.int32) - crop_h, dtype=tf.int32)
        crop_offset_x = tf.random.uniform([], 0, tf.cast(w, tf.int32) - crop_w, dtype=tf.int32)

        image = tf.image.crop_to_bounding_box(image, crop_offset_y, crop_offset_x, crop_h, crop_w)

        keypoints_xy = keypoints[:, :2]
        keypoints_xy = keypoints_xy - tf.cast([crop_offset_x, crop_offset_y], tf.float32)

        x_in_bounds = tf.logical_and(keypoints_xy[:, 0] >= 0, keypoints_xy[:, 0] <= tf.cast(crop_w, tf.float32))
        y_in_bounds = tf.logical_and(keypoints_xy[:, 1] >= 0, keypoints_xy[:, 1] <= tf.cast(crop_h, tf.float32))
        in_bounds = tf.logical_and(x_in_bounds, y_in_bounds)

        visibility = tf.cast(in_bounds, tf.float32) * keypoints[:, 2]
        keypoints = tf.concat([keypoints_xy, tf.expand_dims(visibility, axis=-1)], axis=-1)

        return image, keypoints


    def random_brightness(self, image, keypoints):
        image = tf.image.random_brightness(image, max_delta=self.max_delta)
        return image, keypoints

    def rescale_image(self, image, keypoints, height, width):
        # Handle the shape of the images
        original_size = tf.cast(tf.shape(image), tf.float32)
      
        image = tf.image.resize(image, [height, width])

        # Compute the scale factor for resizing keypoints
        scale = tf.cast([width, height], tf.float32) / (original_size[1], original_size[0])

        # Resize keypoints by scaling y and x coordinates and normalize them
        keypoints_xy = (keypoints[:, :2] * scale)
        keypoints = tf.concat([keypoints_xy, keypoints[:, 2:]], axis=-1)


        return image, keypoints