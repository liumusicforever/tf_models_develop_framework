import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest

import tensorflow as tf
import numpy as np
import numpy.testing as npt

class TestSimpleMlpCase(unittest.TestCase):
    def test_build_graph(self):
        # load simple mlp model;
        from networks.mlp_classifier.graph import model_fn
        from networks.mlp_classifier.params import network_params
        # data,label
        inputs = tf.placeholder("float",[None,224,224,3])
        labels = tf.placeholder("int32", [10])
        
        model = model_fn(inputs,labels,mode = "Train" , params = network_params)

        self.assertTrue(model)
    def test_evaluation(self):
        # load simple mlp model;
        from networks.mlp_classifier.graph import model_fn
        from networks.mlp_classifier.data import create_tfrecord , input_fn
        from networks.mlp_classifier.params import network_params
        from lib.utils import get_all_img_path_label
        
        # data,label
        inputs = tf.placeholder("float",[None,224,224,3])
        labels = tf.placeholder("int32", [None])
        
        # create model graph
        model = model_fn(inputs,labels,mode = tf.estimator.ModeKeys.EVAL,params = network_params)
                
        print ([a for a in model])
        eval_metric_ops = model[4]
        predictions_ops = model[1]
        
        init_op = tf.global_variables_initializer()
        init_lo = tf.local_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init_op)
        sess.run(init_lo)

        
        img = np.zeros((5,224,224,3))
        label_ori = [1 , 0 , 0 , 1 , 1]
        
        feed_dict = {
            inputs : img,
            labels : label_ori
        }
        
        for i in range(3):
            class_tensor = predictions_ops["classes"]
            accuracy_tensor = eval_metric_ops['accuracy']
            accuracy = sess.run([class_tensor , accuracy_tensor],
                    feed_dict = feed_dict)
        
        self.assertAlmostEqual(accuracy[1][0] , 0.4)

    def test_freeze_graph(self):
        '''
        This case for testing is same or not prediction from .ckpt and .pb.
        The test case have three step :
            step 1 : create ckpt with gloval vars init , 
                     and predict output of ckpt from feeding random array
            step 2 : freeze .ckpt to .pb by calling freeze_graph.py
            step 3 : load .pb graph , 
                     and predict .pb output with random array genering from step 1
            step 4 : check if ckpt output equal to pb output

        '''
        tf.reset_default_graph()
        # load simple mlp model;
        import networks.mlp_classifier.graph as graph
        import networks.mlp_classifier.params as params
        import inference
        
        # data,label
        inputs = tf.placeholder("float",[None,224,224,3] , name = "input_op")

        # create model graph
        model = graph.model_fn(inputs,labels = None,mode = tf.estimator.ModeKeys.PREDICT,params = params.network_params)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            # init vars
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            predictions_ops = model[1]['probabilities']

            test_batch = [np.random.rand(224,224,3)]
            feed_dict = {inputs : test_batch}
            ckpt_out = sess.run(predictions_ops , feed_dict = feed_dict)
            ckpt_out = np.array(ckpt_out)

            save_path = saver.save(sess, "/tmp/model/model.ckpt")
            tf.train.write_graph(sess.graph, '/tmp/model', 'graph.pbtxt')

        tf.reset_default_graph()

        # freeze ckpt to pb
        os.system("python3 freeze_graph.py \
                         --input_graph /tmp/model/graph.pbtxt \
                         --input_checkpoint /tmp/model/model.ckpt \
                         --output_graph /tmp/model/model.pb \
                         --output_node_names softmax_tensor")
        tf.reset_default_graph()

        # load pb model , and get input(x) , output(y)
        freeze_model_path = '/tmp/model/model.pb'
        pb_graph , x , y = inference.load_graph(freeze_model_path,
                                input_name = "prefix/input_op:0",
                                output_name = "prefix/softmax_tensor:0")
        
        # forward pb graph with test_batch
        with tf.Session(graph=pb_graph ,config=config) as sess:
            pb_out = sess.run(y, feed_dict={
                x: test_batch
            })
        
        
        # check if ckpt output equal to pb output
        npt.assert_array_equal(pb_out, ckpt_out)
        os.system("rm -rf /tmp/model/")
        

if __name__ == "__main__":
    unittest.main()
    