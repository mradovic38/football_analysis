import json
import os
import tensorflow as tf

def coco_to_tfrecord_converter(self, json_fpath, img_fpath, output_path):
    '''
    Converts a COCO JSON file to a TFRecord file.

    Args:
        json_fpath (str): The file path to the COCO JSON file.
        img_fpath (str): The file path to the directory containing the image files.

    Returns:
        None. The function writes a TFRecord file to the current directory.
    '''

    # Load COCO JSON file
    with open(json_fpath, 'r') as f:
        coco_data = json.load(f)

    # Initialize the class_labels dict
    class_labels = {}

    # Extract category information from COCO JSON
    categories = coco_data.get('categories', [])
    for category in categories:
        class_labels[category['name']] = category['id']

    print(class_labels)

    # Convert COCO JSON to TFRecord
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_info in coco_data['images']:
            # Get the image ID
            image_id = image_info['id']
            # Get annotations for the current image
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            # Generate a TF Example
            tf_example = _create_tf_example(class_labels, image_info, annotations, img_fpath)
            # Write the TF Example to the TFRecord file
            writer.write(tf_example.SerializeToString())

    print(f'TFRecord file saved at {output_path}')

def _create_tf_example(class_labels, image_info, annotations, img_fpath):
    '''
    Creates a TensorFlow Example for an image and its annotations.

    Args:
        class_labels (dict): A dictionary mapping class names to class IDs.
        image_info (dict): A dictionary containing image metadata, including:
            - 'id' (int): The unique identifier for the image.
            - 'file_name' (str): The filename of the image.
            - 'width' (int): The width of the image in pixels.
            - 'height' (int): The height of the image in pixels.
        annotations (list of dict): A list of dictionaries containing annotation data for the image, where each dictionary includes:
            - 'keypoints' (list of float): A list of keypoints for the object, where each keypoint is represented by [x, y, visibility].
        img_fpath (str): The file path to the directory containing the image files.

    Returns:
        tf.train.Example: A TensorFlow Example containing the image and its annotations.
    '''

    # Extract image ID and file path
    image_id = image_info['id']
    image_file = os.path.join(img_fpath, image_info['file_name'])

    # Read the image file
    with tf.io.gfile.GFile(image_file, 'rb') as fid:
        encoded_image = fid.read()

    # Extract image dimensions
    width = image_info['width']
    height = image_info['height']

    # Convert filename to UTF-8 bytes
    filename = image_info['file_name'].encode('utf8')
    image_format = b'jpg'

    # Initialize the keypoints list
    keypoints = []
    # Extract keypoints from annotations
    # Assuming there's only one annotation per image
    keypoints = annotations[0]['keypoints']

    # Create a dictionary of features
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/keypoints': tf.train.Feature(float_list=tf.train.FloatList(value=keypoints))
    }

    # Create a TF Example from the features
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example
