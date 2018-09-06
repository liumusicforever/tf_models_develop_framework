import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import cv2
import numpy as np
import tensorflow as tf 

import lib.utils as utils
from dataset.pascal_voc import parser
from networks.yolov2.region_layer import lastlayer2detection

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'models/taiwan_all_imgaug/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'networks', 'networks/yolov2', 'The path of the network.')

tf.app.flags.DEFINE_string(
    'input_root', 'path to images', 'path to images.')

tf.app.flags.DEFINE_string(
    'output_root', 'path to output', 'path to output.')

tf.app.flags.DEFINE_string(
    'image_size', None, 'network size')

tf.app.flags.DEFINE_string(
    'thresh', None, 'thresh size')



FLAGS = tf.app.flags.FLAGS

def main():
    
    image_list = parser(FLAGS.input_root ,100)
    # image_list = [os.path.join(FLAGS.input_root,i) for i in os.listdir(FLAGS.input_root)]
    
    mod_graph , data_iter , params = utils.load_network(FLAGS.networks)
    
    
    # Define the model
    tf.logging.info("Creating the model...")    
    image_size = int(FLAGS.image_size)
    image = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[None, image_size,
                                        image_size, 3])
    
    net_tensors  = mod_graph.model_fn(image,None,tf.estimator.ModeKeys.PREDICT,params)
    
    mode , pred_ops = net_tensors[0:2]
    
    variables_to_restore = slim.get_model_variables()
    print (variables_to_restore)
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver = tf.train.Saver() 
    saver.restore(sess, checkpoint_path) 
    
    for image_path , labels in image_list:
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size,image_size))
        img = img / 255.0
        
        feed_dict = {image : [img]}
        
        net_out = sess.run(pred_ops , feed_dict = feed_dict)
        
        box_out = lastlayer2detection(net_out , thresh = float(FLAGS.thresh))
        bboxes = box_out[0]
        
        # print (net_out[...,4]) 
        for b in bboxes:
            color = (random.randint(0, 1),random.randint(0, 1),random.randint(0, 1))
            cv2.rectangle(img,(int(b[1]*image_size),int(b[2]*image_size)),(int(b[3]*image_size),int(b[4]*image_size)),color,2)
            clss = list(params.classes_mapping.keys())[list(params.classes_mapping.values()).index(int(b[0]))]
            cv2.putText(img,clss,(int(b[1]*image_size),int(b[2]*image_size)),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            
            
           
        
        cv2.imshow('img',img)
        cv2.waitKey(0)   
        
        
        
    


    

def search_by_file_type(path , file_types):
    file_list = []
    for root , subdir , files in os.walk(path):
        for each_file in files:
            for file_type in file_types:
                if file_type in each_file:
                    image_path = os.path.join(root,each_file)
                    file_list.append(image_path)
    return file_list

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    main()