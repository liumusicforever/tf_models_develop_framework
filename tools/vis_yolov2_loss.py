
"""Visualization Training Process"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import argparse

import cv2
import numpy as np
import tensorflow as tf

import lib.utils as utils
import networks.yolov2.region_layer as region_layer
import networks.yolov2.params as params

example_text = '''example:

python3 tools/vis_yolov2_loss.py --log_dir experiments/base_model --data_dir data/mnist
python3 tools/vis_yolov2_loss.py --log_dir experiments/yolov2 --data_dir /root/data/VOCdevkit/VOC2012/ --networks networks/yolov2/
'''

def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    

    parser.add_argument('--log_dir', default='experiments/mlp_classifier',
                        help="Experiment training result or evaluation result")
    parser.add_argument('--networks',default='networks/mlp_classifier',
                        help="network graph , training confing and data preprocessing")
    parser.add_argument('--data_dir', default='tests/data/',
                        help="Directory containing the dataset")
    
    args = parser.parse_args()

    return args

def draw_detection(img,bboxes):
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    w,h = img.shape[:2]
    color = (random.randint(0, 1),random.randint(0, 1),random.randint(0, 1))
    for b in bboxes:
        b[1:] = [min(i,1) for i in b[1:]]
        b[1:] = [max(i,0) for i in b[1:]]
        cv2.rectangle(img,(int(b[1]*w),int(b[2]*h)),(int(b[3]*w),int(b[4]*h)),color,5)
        clss = list(params.classes_mapping.keys())[list(params.classes_mapping.values()).index(int(b[0]))]
        cv2.putText(img,clss,(int(b[1]*w),int(b[2]*h)),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    
    
    
    return img
    

def vis_loss_trand():
    args = parse_args()
    mod_graph , data_iter , params = utils.load_network(args.networks)

    # create loss graph
    image_op = tf.placeholder("float", [None,None,None,3]) 
    label_op = tf.placeholder("float", [None,None,None,9,None])
    # pred_op = tf.placeholder("float", [None,19,19,5,25])
    # loss_op , _ = mod_graph.yolov2_loss(pred_op,label_op)
    mod = mod_graph.model_fn(image_op,label_op,
                            tf.estimator.ModeKeys.TRAIN,
                            params.network_params)
    pred_op , loss_op , train_op = mod[1:4]
    
    

    # load model into gpu
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    if params.pre_trained:
        sess.run(tf.global_variables_initializer())
        exclude = ['darknet19/conv7_1']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, params.pre_trained)
    else:
        sess.run(tf.global_variables_initializer())
    

    # got data iter
    dataset = data_iter.input_fn(args.data_dir, batch_size = 30 , record_name = 'tfrecord')
    
    
    while True:
        try:
            iterator = dataset.make_one_shot_iterator()
            one_element = iterator.get_next()
            dump_pior = 0
            while True:    
                batch , label = sess.run(one_element)

                feed_dict = {image_op:batch ,label_op:label}
                pred , loss , _ = sess.run(
                    [pred_op , loss_op , train_op],
                    feed_dict = feed_dict)

                print (pred[...,0,0,0,4])
                true_boxes = region_layer.lastlayer2detection(label)
                pred_boxes = region_layer.lastlayer2detection(pred)

                img = batch[0]
                img = draw_detection(img,true_boxes[0])
                img = draw_detection(img,pred_boxes[0])


                true_boxes = np.array(true_boxes)
                # det2[0][0][4] = det2[0][0][4] * 0.9
                # det2[0][0][1] = det2[0][0][1] * 1.01

    #             label2 = region_layer.detection2lastlayer(true_boxes,(19,19))
    #             true_boxes2 = region_layer.lastlayer2detection(label2)
    #             img = draw_detection(img,true_boxes[0])
    #             img = draw_detection(img,true_boxes2[0])


                # matching_boxes = label[..., 0:4]
                # pred_boxes = label2[..., 0:4]

                # coord_loss_wh = np.sum(np.square(np.sqrt(matching_boxes[...,2:4]) - np.sqrt(pred_boxes[...,2:4])))


                # feed_dict = {pred_op:label2 ,label_op:label}
                # loss = sess.run(loss_op , feed_dict = feed_dict)
                # print (loss)

                # # quit()


                if dump_pior % 2 == 0:
                    img = cv2.resize(img,(352,352))
                    cv2.imshow('img',img)
                    cv2.waitKey(10)   
                dump_pior += 1

        except tf.errors.OutOfRangeError:
            pass
    



    
if __name__ == "__main__":
    vis_loss_trand()
