
import argparse 
import tensorflow as tf
 
def load_graph(frozen_graph_filename , input_name , output_name):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
 
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    
    for op in graph.get_operations():
        print(op.name,op.values())

    x = graph.get_tensor_by_name(input_name)
    y = graph.get_tensor_by_name(output_name)

    
    
    return graph , x , y



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", 
                default="/tmp/model/model.pb",
                type=str, 
                help="Frozen model file to import")
    parser.add_argument("--input_name", 
                default="", type=str, 
                help="Name of input tensor.")
    parser.add_argument("--output_name", 
                default="", 
                type=str, 
                help="Name of input tensor.")

    args = parser.parse_args()
    # load pb
    graph , x , y = load_graph(args.frozen_model_filename , args.input_name , args.output_name)

    # TODO : Done the general inference exameple here.
    # with tf.Session(graph=graph) as sess:
    #     y_out = sess.run(y, feed_dict={
    #         x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
    #     })
    #     print(y_out) # [[ 0.]] Yay!
