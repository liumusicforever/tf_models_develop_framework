from __future__ import print_function

import os
import cv2
import tensorflow as tf


# binary data type
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# integer data type
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# float data type
def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(rec_path , dataset , aug = None):
    '''
    dataset : (num_example , [path , label])
    '''
    # create tf record writer
    writer = tf.python_io.TFRecordWriter(rec_path)
    
    for i,(image_filename, label) in enumerate(dataset):
        image = load_image(image_filename)
        
        height, width, depth = image.shape

        image_string = image.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(height),
          'width': _int64_feature(width),
          'image_string': _bytes_feature(image_string),
          'label': _float32_feature([label])}))

        writer.write(example.SerializeToString())
        print('processing {}/{}'.format(i,len(dataset)), end='\r')
    writer.close()


def load_image(path):
#     print (path)
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(path)
    if img is None:
        return None
    
    
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def parser(record):
    keys_to_features = {
        "image_string": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.float32)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_string"], tf.uint8)
    
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[224, 224, 3])
    image = image / 255.0
    label = tf.cast(parsed["label"], tf.int32)
    
    return image, label


def input_fn(data_dir ,
             batch_size,
             num_epochs = 1 ,
             record_name = 'tfrecord',
             is_shuffle = False):
    filenames = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if record_name in file]
    if len(filenames) == 0:
        raise ("your input data_dir is not inclue record file ")
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser,num_parallel_calls = 8)
    dataset = dataset.repeat(num_epochs)  # repeat for multiple epochs
        
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    if is_shuffle:
        dataset = dataset.shuffle(3)

    return dataset


    