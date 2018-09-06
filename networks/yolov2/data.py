import os 
import random

import cv2
import numpy as np
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

import networks.yolov2.params as params
import networks.yolov2.region_layer as region_layer

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
    dataset : (num_example , [path , bboxes])
    bounding boxes format:
        [clss , xmin_p , ymin_p , xmax_p , ymax_p]
    '''
    # create tf record writer
    writer = tf.python_io.TFRecordWriter(rec_path)
    
    if aug:
        # defind data augmentation pipe line
        aug = iaa.Sequential([
            iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.15)), # random crops
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.Affine(
                scale={"x": (0.6, 1), "y": (0.6, 1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                )
        ])
    
    for i,(image_filename, label) in enumerate(dataset):
        
        image , label = load_image(image_filename,label,aug)
        image_string = image.tostring()
        
        height, width, depth = image.shape
        
        label = labelname2clssid(label)
        # (num_bboxes , 5)
        label = np.array(labelpadding(label),dtype=np.float32)
        # (200,5)
        label_string = label.tostring()
        
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_string': _bytes_feature(image_string),
                'label_string': _bytes_feature(label_string),
                'height': _int64_feature(height),
                'width': _int64_feature(width)
            }
        ))

        writer.write(example.SerializeToString())
        print('processing {}/{}'.format(i,len(dataset)), end='\r')
    writer.close()


def labelname2clssid(label):
    for i,box in enumerate(label):
        clss_name = box[0]
        clss_id = params.classes_mapping['person']
        label[i][0] = clss_id
    return label

def labelpadding(label,max_num_boxes = 200):
    new_label = []
    for i,box in enumerate(label):
        new_label.append(box)
    for i in range(max_num_boxes - len(new_label)):
        new_label.append([0.0,0.0,0.0,0.0,0.0])
        
    return new_label
def load_image(path,label,aug):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(path)
    height, width, depth = img.shape
    
    if not aug:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (720,720))
        xmin , ymin ,xmax ,ymax = label[1:]
        xmin = min(max(xmin,0),0.999)
        ymin = min(max(ymin,0),0.999)
        xmax = max(min(xmax,0.999),0)
        ymax = max(min(ymax,0.999),0)
        label = [label[i][0],xmin,ymin,xmax,ymax]
        return img , label
    else:
        # insert boxes location into imgaug
        box_axis = [ia.BoundingBox(
                        x1=b[1]*width, 
                        y1=b[2]*height, 
                        x2=b[3]*width, 
                        y2=b[4]*height) for b in label]
        bbs = ia.BoundingBoxesOnImage(box_axis , shape=img.shape)
        
        seq = aug
        
        seq_det = seq.to_deterministic()
        
        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        
        h, w, d = img.shape
        label_aug = []
        for i in range(len(bbs_aug.bounding_boxes)):
            box = bbs_aug.bounding_boxes[i]
            clss = label[i][0]
            xmin = min(max(box.x1/w,0),0.999)
            ymin = min(max(box.y1/h,0),0.999)
            xmax = max(min(box.x2/w,0.999),0)
            ymax = max(min(box.y2/h,0.999),0)
            label_aug.append([clss , xmin, ymin , xmax , ymax])
            
        # for b in label:
        #     print (b)
        #     cv2.rectangle(img,(int(b[1]*width),int(b[2]*height)),(int(b[3]*width),int(b[4]*height)),(55,255,155),5)
            
        # for b in label_aug:
        #     print (b)
        #     cv2.rectangle(image_aug,(int(b[1]*w),int(b[2]*h)),(int(b[3]*w),int(b[4]*h)),(55,255,155),5)
        
        # cv2.imshow('a',img)
        # cv2.imshow('image_aug',image_aug)
        # cv2.waitKey(0)
    if img is None:
        return None
    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
    image_aug = cv2.resize(image_aug, (720,720))

    return image_aug , label_aug

def parser(record):
    
    keys_to_features = {
        "image_string": tf.FixedLenFeature([], tf.string),
        "label_string":tf.FixedLenFeature([], tf.string),
        'height':    tf.FixedLenFeature([], tf.int64),
        'width':    tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    
    H= tf.cast(parsed['height'], tf.int32)
    W = tf.cast(parsed['width'], tf.int32)
    
    # decode image
    image = tf.decode_raw(parsed["image_string"], tf.uint8)
    image = tf.cast(image, tf.float32)
    # reshape image
    image = tf.reshape(image, 
                       shape=tf.stack([H, W, 3]))
    image = image / 255.0
    

    label = tf.decode_raw(parsed["label_string"], tf.float32)
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, 
                       shape=[200,5])
    
    return image, label

def resize_in_batch(image , label):
    in_size = random.choice(params.training_scale)
    print ('training with size : {}'.format(in_size))
    
    new_image = []
    for img in image:
        img = cv2.resize(img,(in_size,in_size))
        new_image.append(img)

    label_size = int(in_size/32)    
    
    label = region_layer.detection2lastlayer(
                        label , 
                        out_shapes = (label_size,label_size))
    label = np.array(label,dtype = np.float32)

    return new_image , label
    

def input_fn(data_dir ,
             batch_size,
             num_epochs = 1 ,
             record_name = 'tfrecord',
             is_shuffle = False):
    filenames = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if record_name in file]
    if len(filenames) == 0:
        raise ("your input data_dir is not inclue record file ")
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser,num_parallel_calls = 16)
    dataset = dataset.repeat(num_epochs)  # repeat for multiple epochs
    
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    
    
    

    if is_shuffle:
        dataset = dataset.shuffle(20)
    
    # Mutiple scale during training
    dataset = dataset.map(
                lambda image , label: 
                tuple(tf.py_func(resize_in_batch, 
                      [image , label],
                      [tf.float32, tf.float32])),
                num_parallel_calls=32).prefetch(32)

    return dataset

