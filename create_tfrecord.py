import os
import argparse
import random

import tensorflow as tf


import lib.utils as utils

example_text = '''example:

python3 create_tfrecord.py --networks networks/mlp_classifier --data_dir data/mnist --dataset dataset/classification.py
python3 create_tfrecord.py --networks networks/yolov2/ --data_dir /root/data/VOCdevkit/VOC2012/ --dataset dataset/pascal_voc.py --split 5 --record_name train.tfrecord

'''


def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--networks',default='networks/mlp_classifier',
                        help="network graph , training confing and data preprocessing")
    parser.add_argument('--data_dir', default='tests/data/',
                        help="Directory containing the dataset")
    
    parser.add_argument('--dataset', default='',
                        help="Path to dataset loader")
    
    parser.add_argument('--imgaug', default='',
                        help="Create with data augumentation")
    
    parser.add_argument('--split', default='5',
                        help="How many number of spliting dataset.")

    parser.add_argument('--record_name', default='train.tfrecord',
                        help="Name of the records")
    
    args = parser.parse_args()

    return args

def example_create_tfrecord():
    
    args = parse_args()
    
    mod_graph , data_iter , params = utils.load_network(args.networks)
    datset = utils.load_module(args.dataset)
    
    # source classification root
    data_root = args.data_dir
    # tfrecord file path
    rec_path = os.path.join(data_root , args.record_name)
    
    data_list = datset.parser(data_root)
    
    
    random.shuffle(data_list)
    
    # split dataset into number of N pack
    for i in range(int(args.split)):
        split_start = int(len(data_list)*i/int(args.split))
        split_end = int(len(data_list)*(i+1)/int(args.split))
        split_list = data_list[split_start:split_end]
        rec_name = '{}.{}'.format(rec_path,i)
        data_iter.create_tfrecord(rec_name,split_list)
        print('processing {}/{}'.format(split_end,len(data_list)))
        
        
import  networks.yolov2.region_layer as region_layer
import cv2

    
def example_read_tfrecord():
    args = parse_args()
    data_root = args.data_dir
    mod_graph , data_iter , params = utils.load_network(args.networks)
    # get tf.datset from root of tfrecord files
    dataset = data_iter.input_fn(data_root, batch_size = 3 , record_name = 'tfrecord')
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    try:
        with tf.Session() as sess:
            while True:
                img , label = sess.run(one_element)
                img = img[0]
                prediction = region_layer.lastlayer2detection(label , 0.6)
                print (prediction)
                h , w  = img.shape[:2]
                

                for b in prediction[0]:
                    cv2.rectangle(img,(int(b[2]*w),int(b[3]*h)),(int(b[4]*w),int(b[5]*h)),(55,255,155),5)
                img = cv2.resize(img, (416,416))


                cv2.imshow('img',img)
                cv2.waitKey(0)
                
                
    except tf.errors.OutOfRangeError:
        pass


if __name__ == "__main__":
    example_create_tfrecord()
    # example_read_tfrecord()