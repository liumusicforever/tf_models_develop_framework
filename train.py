"""Train the model"""

import argparse
import os

import tensorflow as tf

import lib.utils as utils

example_text = '''example:

python3 train.py --log_dir experiments/base_model --data_dir data/mnist
python3 train.py --log_dir experiments/yolov2 --data_dir /root/data/VOCdevkit/VOC2012/ --networks networks/yolov2/
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
    
    
    
def main():
    tf.reset_default_graph()
    
    tf.set_random_seed(1234)
    tf.logging.set_verbosity(tf.logging.INFO)
    
    args = parse_args()
    
    
    mod_graph , data_iter , params = utils.load_network(args.networks)
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=1234,
                                    model_dir=args.log_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    session_config = sess_config)
    
    
    estimator = tf.estimator.Estimator(
#                     model_fn=tf.contrib.estimator.replicate_model_fn(mod_graph.model_fn),
                    model_fn=mod_graph.model_fn,
                    params = params.network_params ,
                    config=config,
                )
    
#     train_input_fn = lambda: data_iter.input_fn(args.data_dir,
#                         params.batch_size,
#                         params.num_epochs,
#                         record_name = "train.tfrecord",
#                         is_shuffle = True)
    
#     eval_input_fn = lambda: data_iter.input_fn(args.data_dir,
#                         params.batch_size,
#                         record_name = "val.tfrecord",
#                         is_shuffle = False)
    
#     train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000000000)
#     eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
#     tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    
#     # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: data_iter.input_fn(args.data_dir,
                                               params.batch_size,
                                               params.num_epochs,
                                               record_name = "train.tfrecord",
                                               is_shuffle = True))
    
#     # Evaluate the model on the test set
#     tf.logging.info("Evaluation on test set.")
    estimator.evaluate(lambda: data_iter.input_fn(args.data_dir,
                                                  params.batch_size,
                                                  record_name = "val.tfrecord",
                                                  is_shuffle = False))
    
if __name__ == "__main__":
    main()

