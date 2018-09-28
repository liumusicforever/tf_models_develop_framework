'''
This script used to generate features which could evaluate from the repo below:
https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate_gpu.py

'''

import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import scipy.io
import numpy as np
import tensorflow as tf

import lib.utils as utils
import dataset.classification as dataset


example_text = '''example:

python3 tools/generate_facenet_feature.py --log_dir experiments/facenet_inception_resnet_v1 --data_dir /root/data/

'''



def parse_args():
    
    parser = argparse.ArgumentParser( epilog=example_text,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)    

    parser.add_argument('--log_dir', default='experiments/facenet_inception_resnet_v1',
                        help="Experiment training result or evaluation result")
    parser.add_argument('--networks',default='networks/facenet_inception_resnet_v1',
                        help="network graph , training confing and data preprocessing")
    parser.add_argument('--data_dir', default='/root/data/',
                        help="Directory containing the dataset")
    
    args = parser.parse_args()

    return args

def gen_feat(img_list,image_op,sess,embeddings):
    feat_list = []
    clss_id_list = []
    cam_id_list = []
    for i,(img_path,clss_id,cam_id) in enumerate(img_list):
        print ('{}/{}'.format(i,len(img_list)))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        feats = sess.run(embeddings,feed_dict = {image_op : [img]})
        feat_list.append(feats[0])
        clss_id_list.append(clss_id)
        cam_id_list.append(cam_id)
    
    feat_list = np.stack(feat_list)
    
    
    print (feat_list.shape)

    
    return feat_list,clss_id_list,cam_id_list
        

def search_by_file_type(path , file_types):
    file_list = []
    for root , subdir , files in os.walk(path):
        for each_file in files:
            for file_type in file_types:
                if file_type in each_file:
                    image_path = os.path.join(root,each_file)
                    clss_id = each_file.split('_')[0]
                    cam_id = each_file.split('_')[1].split('s')[0].split('c')[1]
                    file_list.append([image_path,int(clss_id),int(cam_id)])
    return file_list

def main():
    args = parse_args()
    mod_graph , data_iter , params = utils.load_network(args.networks)

    # create loss graph
    image_op = tf.placeholder("float", [None,160,160,3]) 
    label_op = None

    mod = mod_graph.model_fn(image_op,label_op,
                            tf.estimator.ModeKeys.PREDICT,
                            params.network_params)
    
    
    embeddings = mod[1]['embeddings']
    
    
    
    if tf.gfile.IsDirectory(args.log_dir):
        checkpoint_path = tf.train.latest_checkpoint(args.log_dir)
    else:
        checkpoint_path = args.log_dir

    # load model into gpu
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    if params.pre_trained:
        exclude = []
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude= exclude)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    print ('finished restore : {}'.format(checkpoint_path))
    
    query_list = search_by_file_type(os.path.join(args.data_dir,'query'),[".jpg"])
    query_feats,query_labels,query_cam_ids = gen_feat(query_list,image_op,sess,embeddings)
    
    gallery_list = search_by_file_type(os.path.join(args.data_dir,'gallery'),[".jpg"])
    gallery_feats,gallery_labels,gallery_cam_ids = gen_feat(gallery_list,image_op,sess,embeddings)
    
    # Save to Matlab for check
    result = {'gallery_f':gallery_feats,'gallery_label':gallery_labels,'gallery_cam':gallery_cam_ids,'query_f':query_feats,'query_label':query_labels,'query_cam':query_cam_ids}
    scipy.io.savemat('pytorch_result.mat',result)
    
    
    
if __name__ == "__main__":
    main()