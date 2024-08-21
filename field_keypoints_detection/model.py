import tensorflow as tf
from dataset_utils import get_dataset
from field_keypoints_detection.data_preprocessing import ResizeAndNormalize, Augmentation
from field_keypoints_detection.loss import visibility_based_mse
from field_keypoints_detection.metrics import VisibilityAccuracy, VisibleKeypointMSE

class KeypointModel:
    def __init__(self, train_tfrecord_path, val_tfrecord_path, test_tfrecord_path, num_keypoints, batch_size=32):
        self.num_keypoints = num_keypoints
        self.batch_size = batch_size

        # Initialize preprocessing and augmentation
        self.resize_and_normalize = ResizeAndNormalize(target_size=(300, 300))
        self.augmentation = Augmentation(rescale_size=(300, 300))

        # Create datasets
        self.train_dataset = self.load_and_preprocess_dataset(train_tfrecord_path, training=True)
        self.val_dataset = self.load_and_preprocess_dataset(val_tfrecord_path)
        self.test_dataset = self.load_and_preprocess_dataset(test_tfrecord_path)


    def load_and_preprocess_dataset(self, tfrecord_path, training=False):
        dataset = get_dataset(tfrecord_path, self.batch_size)
        dataset = dataset.map(lambda image, keypoints: self.preprocess(image, keypoints, training=training),
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def preprocess(self, image, keypoints, training=False):
        
        # Reshape
        keypoints_shape = tf.shape(keypoints)
        if len(keypoints_shape) == 2:
          keypoints = tf.expand_dims(keypoints, axis=0)  # Add batch dimension if missing

        # Reshape keypoints to ensure they have the shape (batch_size, 32, 3)
        keypoints = tf.reshape(keypoints, [-1, 32, 3])

        # Apply augmentation (only during training)
        if training:
            image, keypoints = self.augmentation.apply(image, keypoints)

        # Apply resizing and normalization
        image, keypoints = self.resize_and_normalize.apply(image, keypoints)

        return image, keypoints

    def build(self):
        # Define input layers
        image_input = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)

        x = tf.keras.applications.efficientnet.preprocess_input(image_input)

        # Use EfficientNetB3 as feature extractor
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=image_input)
        base_model.trainable = False  # Freeze the base model

        x = base_model(x)

        # Add custom layers on top
        x = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(self.num_keypoints * 3, activation='linear')(x)
        output = tf.keras.layers.Reshape((self.num_keypoints, 3))(output)

        # Create the model
        self.model = tf.keras.Model(inputs=[image_input], outputs=output)

        # Compile the model
        self.model.compile(optimizer='adam', loss=visibility_based_mse(),
                          metrics=[VisibleKeypointMSE(), VisibilityAccuracy()])

    def train(self, epochs=10):
        self.training = True
        # Train the model
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset)

    def evaluate(self):
        self.training = False
        # Evaluate the model
        self.model.evaluate(self.test_dataset)

