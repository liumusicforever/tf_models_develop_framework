
import os

import imp

def load_module(module_path):
    assert os.path.exists(module_path) ,\
            'Not found error : {}'.format(module_path)
    mod_graph = imp.load_source('',module_path)
    return mod_graph
    
def load_network(network_path):
    
    graph_path = os.path.join(network_path,'graph.py')
    iter_path = os.path.join(network_path,'data.py')
    params_path = os.path.join(network_path,'params.py')
    
    assert os.path.exists(graph_path) ,\
            'you need to define <network name>/graph.py'
        
    assert os.path.exists(iter_path) ,\
            'you need to define <network name>/data.py'
        
    assert os.path.exists(params_path) ,\
            'you need to define <network name>/params.py'
    
        
    
    mod_graph = imp.load_source('graph',graph_path)
    data_iter = imp.load_source('data_iter',iter_path)
    params = imp.load_source('params',params_path)
    
    
    
    assert 'model_fn' in dir(mod_graph)
    assert 'input_fn' in dir(data_iter)
    
    return mod_graph , data_iter , params