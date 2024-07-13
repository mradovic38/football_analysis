import tensorflow as tf
from dataset_utils import load_tfrecords
from keypoint_data_augmentation import ResizeAndNormalizeLayer, AugmentationLayer
from keypoint_loss import visibility_based_mse

class KeypointModel:
    def __init__(self, train_tfrecord_path, val_tfrecord_path, test_tfrecord_path, batch_size=32):
        # Loading TFRecords
        train_dataset =  load_tfrecords(train_tfrecord_path)
        val_dataset =  load_tfrecords(val_tfrecord_path)
        test_dataset = load_tfrecords(test_tfrecord_path)

        # Batching
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        # Prefetching
        train_dataset =  train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset =  val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset =  test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def build(self):
        input_image = tf.keras.layers.Input(shape=(None, None, 3))
        input_keypoints = tf.keras.layers.Input(shape=(None, 3))  # Use None for the number of keypoints

        # Get the number of keypoints dynamically
        num_keypoints = tf.shape(input_keypoints)[1]

        # Apply resize and normalization layer
        resized_image, normalized_keypoints = ResizeAndNormalizeLayer(target_size=(300, 300))([input_image, input_keypoints])

        # Apply augmentation layer (only during training)
        augmented_image, augmented_keypoints = AugmentationLayer()([resized_image, normalized_keypoints])

        # Define the rest of the model architecture
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(augmented_image)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(num_keypoints * 3, activation='linear')(x)
        output = tf.keras.layers.Reshape((num_keypoints, 3))(output) 

        model = tf.keras.Model(inputs=[input_image, input_keypoints], outputs=[output])

        model.compile(optimizer='adam', loss=visibility_based_mse)

        self.model = model

    def train(self, epochs=10):
        # Train the model
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset)

    def evaluate(self):
        # Evaluate the model
        self.model.evaluate(self.test_dataset)
