import tensorflow as tf

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

def load_tfrecords(fpath):
    raw_dataset = tf.data.TFRecordDataset(fpath)
    parsed_and_decoded_dataset = raw_dataset.map(_parse_and_decode_function, num_parallel_calls=tf.data.AUTOTUNE)
    return parsed_and_decoded_dataset