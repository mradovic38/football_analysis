import tensorflow as tf
from functools import partial

def _parse_and_decode_function(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/keypoints': tf.io.VarLenFeature(tf.float32),
    }
    # Parse the input tf.train.Example proto using the feature description
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the image
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)

    # Get the keypoints
    keypoints = tf.sparse.to_dense(example['image/object/keypoints'])

    return image, keypoints

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(_parse_and_decode_function), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filenames, batch_size):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset