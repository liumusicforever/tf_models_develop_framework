import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import logging

import numpy as np
import numpy.testing as npt
import tensorflow as tf

logging.basicConfig(level = logging.INFO)

class TestYoloV2Case(unittest.TestCase):
#     @unittest.skip("Individual unit tests temporarily")
    def test_data(self):
        ### TODO : finish create tf record testing
        import networks.yolov2.data as data
        import dataset.pascal_voc as dataset
        data_root = '/root/data/VOCdevkit/VOC2012/'
        data_list = dataset.parser(data_root , 10)
        
        data.create_tfrecord('/tmp/tfrecord',data_list , aug = None)
    def test_detection2lastlayer(self):
        import networks.yolov2.region_layer as region_layer
        
        import cv2
        
        # create test label 
        test_img = np.zeros((1,416,416,3))
        test_label = np.array([
            [[4, 0.666, 0.10810810810810811, 0.864, 0.2108108108108109],
             [3, 0.322, 0.3123123123123123, 0.584, 0.6885885885885885]],
            [[1, 0.001, 0.10810810810810811, 0.999, 0.2108108108108109],
             [0, 0.0, 0.0, 0.0, 0.0]]
        ])
        
        lastlayer = region_layer.detection2lastlayer(test_label)
        detection = region_layer.lastlayer2detection(lastlayer)
        
        npt.assert_array_almost_equal(test_label[0][0] , detection[0][1])
        
    def test_yolov2_loss(self):
        import networks.yolov2.region_layer as region_layer
        import networks.yolov2.graph as graph
        
        target = np.array([
            [[17, 0.01, 0.0210810810810811, 0.02, 0.0308108108108109],
             [14, 0.872, 0.86123123123123123, 0.994, 0.9885885885885885]],
            [[0, 0.001, 0.87810810810810811, 0.02, 0.99108108108108109],
             [5, 0.0, 0.0, 0.0, 0.0]]
        ])
        
        # bad prediction
        pred2 = np.array([
            [[17, 0.01, 0.8710810810810811, 0.02, 0.9908108108108109],
             [14, 0.872, 0.86123123123123123, 0.994, 0.9885885885885885]],
            [[0, 0.1, 0.7, 0.02, 0.5],
             [5, 0.0, 0.0, 0.0, 0.0]]
        ])
        
        # perfect prediction
        pred1 = target
        
        lastlayer_target = region_layer.detection2lastlayer(target , (3,3))
        lastlayer_pred1 = region_layer.detection2lastlayer(pred1, (3,3))
        lastlayer_pred2 = region_layer.detection2lastlayer(pred2, (3,3))
        
        target_tensor = tf.cast(lastlayer_target , dtype=tf.float32)
        pred1_tensor = tf.cast(lastlayer_pred1, dtype=tf.float32)
        pred2_tensor = tf.cast(lastlayer_pred2, dtype=tf.float32)
        
        good_exp , _  = graph.yolov2_loss(pred1_tensor , target_tensor)
        bad_exp , _  = graph.yolov2_loss(pred2_tensor , target_tensor)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            good_res = sess.run(good_exp)
            bad_res = sess.run(bad_exp)
            print (good_res)
            print (bad_res)
        self.assertTrue(good_res < bad_res)
        
    def test_yolov2_eval(self):
        import networks.yolov2.region_layer as region_layer
        import networks.yolov2.graph as graph
        
        target = np.array([
            [[17, 0.01, 0.0210810810810811, 0.02, 0.0308108108108109],
             [14, 0.872, 0.86123123123123123, 0.994, 0.9885885885885885]],
            [[0, 0.001, 0.87810810810810811, 0.02, 0.99108108108108109],
             [16, 0.018, 0.0280810810810811, 0.88, 0.880]]
        ])
        
        # bad prediction
        pred2 = np.array([
            [[16, 0, 0, 0, 0],
             [16, 0, 0, 0, 0]],
            [[16, 0, 0, 0, 0],
             [16, 0.019, 0.025, 0.88, 0.88]]
        ])
        
        # perfect prediction
        pred1 = target
        
        lastlayer_target = region_layer.detection2lastlayer(target , (3,3))
        lastlayer_pred1 = region_layer.detection2lastlayer(pred1, (3,3))
        lastlayer_pred2 = region_layer.detection2lastlayer(pred2, (3,3))
        
        target_tensor = tf.cast(lastlayer_target , dtype=tf.float32)
        pred1_tensor = tf.cast(lastlayer_pred1, dtype=tf.float32)
        pred2_tensor = tf.cast(lastlayer_pred2, dtype=tf.float32)
        
        
        acc = graph.yolov2_eval(pred2_tensor , target_tensor)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            
            a = sess.run(acc)
            
            print (a)
            
            
        
        
if __name__ == "__main__":
    unittest.main()
        