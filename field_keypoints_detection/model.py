import tensorflow as tf
from dataset_utils import load_tfrecords
from field_keypoints_detection.data_augmentation import ResizeAndNormalizeLayer, AugmentationLayer
from field_keypoints_detection.loss import visibility_based_mse
from field_keypoints_detection.metrics import VisibilityAccuracy, VisibleKeypointMSE

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
        input_keypoints = tf.keras.layers.Input(shape=(None, 3)) 

        # Get the number of keypoints dynamically
        num_keypoints = tf.shape(input_keypoints)[1]

        # Apply resize and normalization layer
        resized_image, normalized_keypoints = ResizeAndNormalizeLayer(target_size=(300, 300))([input_image, input_keypoints])

        # Apply augmentation layer (only during training)
        augmented_image, augmented_keypoints = AugmentationLayer()([resized_image, normalized_keypoints])

        # Use EfficientNetB3 as feature extractor
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=augmented_image)
        base_model.trainable = False  # Freeze the base model

        # Add custom layers on top
        x = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same')(base_model.output)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(num_keypoints * 3, activation='linear')(x)
        output = tf.keras.layers.Reshape((num_keypoints, 3))(output)  

        model = tf.keras.Model(inputs=[input_image, input_keypoints], outputs=[output])

        model.compile(optimizer='adam', loss=visibility_based_mse,
                      metrics=[VisibleKeypointMSE(), VisibilityAccuracy()])

        self.model = model


    def train(self, epochs=10):
        # Train the model
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset)

    def evaluate(self):
        # Evaluate the model
        self.model.evaluate(self.test_dataset)
