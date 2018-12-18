import uuid

import numpy as np
import tensorflow as tf

import networks.yolov2.params as config

import networks.yolov2.region_layer as region_layer
import metrics.core.standard_fields as standard_fields
import metrics.coco_evaluation as coco_evaluation


slim = tf.contrib.slim


def model_fn(features, labels, mode , params):
    """
        Model function for YOLOv2.
    * Remove the passthrough layers
    * Change leaky-relu to relu
    """
    
    
    with tf.variable_scope('darknet19', [features]) as sc:
        # learning phase
        if mode == tf.estimator.ModeKeys.PREDICT:
            is_training = False
        else:
            is_training = True
        batch_norm_params = {
            'is_training':is_training,
        }
        with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.glorot_normal_initializer(),
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
            
            # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # remove normalizer_fn=slim.batch_norm
            
            features.set_shape([None,None,None,3])
            net = slim.conv2d(features, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv3_2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 256, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv4_2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv5_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            net = slim.conv2d(net, 256, [1, 1], scope='conv5_4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv6_2')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_3')
            net = slim.conv2d(net, 512, [1, 1], scope='conv6_4')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_5')

            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_6')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_7')
        
        if config.pre_trained:
            exclude = ['darknet19/conv7_1']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(config.pre_trained, 
                              {v.name.split(':')[0]: v for v in variables_to_restore})

        with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.glorot_normal_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      activation_fn=tf.nn.relu,
                      biases_initializer = tf.zeros_initializer()):
            net = slim.conv2d(net, config.num_anchors * (5 + config.num_classes),
                              [1, 1], scope='conv7_1',activation_fn=None)
    
    net_dims = tf.shape(net)
    final_dim = tf.stack([-1, net_dims[1], net_dims[2], 
                          config.num_anchors, 5 + config.num_classes])
        
    net = tf.reshape(net , shape=final_dim , name = 'final_conv')
    
    
    box_xy = tf.nn.sigmoid(net[..., :2])
    box_wh = tf.exp(net[..., 2:4])
    box_confidence = tf.nn.sigmoid(net[..., 4:5])
    box_class_probs = tf.nn.softmax(net[..., 5:])
    
    net = tf.concat([box_xy, box_wh , box_confidence , box_class_probs],
                    axis = -1 ,
                    name = 'prediction')
    
    prediction = net
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss , logging_hook = yolov2_loss(prediction , labels)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_vars = tf.trainable_variables()
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["lr"])
#         optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"]) 
        optimizer = tf.train.MomentumOptimizer(learning_rate=params["lr"], momentum=0.9)
    
        # mutiple gpu training require
#         optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        
        grads_and_vars = optimizer.compute_gradients(loss, train_vars)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, 11), var) for grad, var in grads_and_vars]
        
        # batch normalize update op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(
                      clipped_grads_and_vars,
                      global_step = tf.train.get_global_step())

#             train_op = optimizer.minimize(
#                 loss,
#                 global_step=tf.train.get_global_step())
            

        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op,
                                          predictions=prediction,
                                          training_hooks = [logging_hook])
    
    eval_metric_ops = yolov2_eval(prediction , labels)
    
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss,predictions = prediction ,  eval_metric_ops=eval_metric_ops)



def yolov2_loss(pred , target):
    '''
    Parameter :
        pred :
            dtype : tensor
            shape : (batch_size , H , W , num_anchors , (x , y , w , h , prob) + num_clss)
        target :
            dtype : tensor
            shape : (batch_size , H , W , num_anchors , (x , y , w , h , prob) + num_clss)
    return :
        total_loss:
            dtype : tensor
    
    implement ref : https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py
    '''

    target_dims = tf.shape(target)
    pred_dims = pred.get_shape().as_list()
    
    
    detectors_mask = target[..., 4:5]
    matching_classes = target[..., 5:]
    matching_boxes = target[..., 0:4]
    pred_confidence = pred[..., 4:5]
    pred_class_prob = pred[..., 5:]
    pred_boxes = pred[..., 0:4]
    
    # Got Cxy and Pwh
    final_dim = tf.stack([-1, target_dims[1], target_dims[2]])
    Cxy = tf.reshape(tf.range(target_dims[1]* target_dims[2]) , shape = final_dim)
    cx = tf.expand_dims(tf.cast(Cxy % target_dims[2],tf.float32),-1)
    cy = tf.expand_dims(tf.cast(tf.cast(Cxy / target_dims[1] ,tf.int32),tf.float32),-1)
    Cxy = tf.concat([cx,cy],-1) 
    Cxy = tf.expand_dims(Cxy , 3)
    # Cxy : [1, H, W, 1, 2]
    
    anchors = tf.cast(config.anchors , dtype=tf.float32)
    anchors = tf.expand_dims(tf.expand_dims(tf.expand_dims(anchors , 0) , 0) , 0)
    Pwh = anchors
    # Pwh : [1, H, W, 5, 2]
    
    # Find IOU of each predicted box with each ground truth box.
    true_xy = (target[..., 0:2] + Cxy)/ tf.cast(tf.shape(Cxy)[1:3],tf.float32)
    true_wh = target[..., 2:4] * Pwh 
    pred_xy = (pred[..., 0:2] + Cxy)/ tf.cast(tf.shape(Cxy)[1:3],tf.float32)
    pred_wh = pred[..., 2:4] * Pwh 
    
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    
    # compute intersect areas
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / (union_areas + 1e-20)
    
    # Best IOUs for each location.
    best_ious = tf.reduce_max(iou_scores, axis=3)  # Best IOU scores.
    best_ious_dim = tf.stack([target_dims[0] , target_dims[1] , target_dims[2] ,1,1])
    best_ious = tf.reshape(best_ious , shape = best_ious_dim)
    
    # A detector has found an object if IOU > thresh for some true box.
    object_detections = tf.cast(best_ious > 0.6,  tf.float32)
    
    # Determine confidence weights from object and no_object weights.
    no_object_weights = ((1 - object_detections) * (1 - detectors_mask))
    
    no_objects_loss = config.no_object_scale  * tf.square(no_object_weights * pred_confidence)
    
    objects_loss = (config.object_scale * detectors_mask *
                        tf.square(1 - pred_confidence))
    
    confidence_loss = no_objects_loss + objects_loss
    
    # Classification loss for matching detections.
    classification_loss = (config.class_scale * detectors_mask *
                           tf.square(matching_classes - pred_class_prob))
    
    # Coordinate loss for matching detection boxes.
    coord_weight = config.coordinates_scale * detectors_mask
    coord_loss_xy = tf.square(true_xy - pred_xy)
    coord_loss_wh = tf.square(tf.sqrt(true_wh)- tf.sqrt(pred_wh))
    coordinates_loss = coord_weight * (coord_loss_xy + coord_loss_wh)
    
    confidence_loss_sum = tf.reduce_sum(confidence_loss)
    classification_loss_sum = tf.reduce_sum(classification_loss)
    coordinates_loss_sum = tf.reduce_sum(coordinates_loss)
    
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    
    # total_loss = tf.Print(
    #     total_loss, [
    #         total_loss, 
    #         classification_loss_sum, 
    #         coordinates_loss_sum,
    #         confidence_loss_sum,
    #         tf.reduce_sum(objects_loss),
    #         tf.reduce_sum(no_objects_loss)
    #     ],
    #     message='')
    log_info = {"total_loss" : total_loss ,
                "conf_loss" : confidence_loss_sum,
                "class_loss": classification_loss_sum,
                "box_coord_loss":coordinates_loss_sum,
                "objects_loss":tf.reduce_sum(objects_loss),
                "no_objects_loss":tf.reduce_sum(no_objects_loss)}
#                 "detectors_mask":(pred_confidence)
    logging_hook = tf.train.LoggingTensorHook(log_info, every_n_iter=1)
    
    return total_loss , logging_hook

    
    
def yolov2_eval(pred , target , thresh = 0.5):
    '''
    Parameter :
        pred :
            dtype : tensor
            shape : (batch_size , H , W , num_anchors , (x , y , w , h , prob) + num_clss)
        target :
            dtype : tensor
            shape : (batch_size , H , W , num_anchors , (x , y , w , h , prob) + num_clss)
    return :
        miss_rate:
            dtype : tensor
        fp_rate:
            dtype : tensor
    '''
    
    
    
    image_ids,pred_clss,pred_prob,pred_boxes,target_clss,target_boxes = tf.py_func(
                   py_fun_yolo_eval, 
                   inp = [pred , target], 
                   Tout = [tf.string,tf.float64,tf.float64,tf.int64,tf.float64,tf.int64])
    
    category_list = [{'id':str(clss_id) , 'name':clss_name} for clss_name,clss_id in config.classes_mapping.items()]
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list)
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    
    image_ids.set_shape([None])
    eval_dict = {
        input_data_fields.key: image_ids,
        input_data_fields.groundtruth_boxes: target_boxes,
        input_data_fields.groundtruth_classes: target_clss,
        detection_fields.detection_boxes: pred_boxes,
        detection_fields.detection_scores: pred_prob,
        detection_fields.detection_classes: pred_clss
    }
    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)



    return  eval_metric_ops
    
    
    
def py_fun_yolo_eval(pred , target, thresh = 0.5):
    
    
    pred_boxes = region_layer.lastlayer2detection(pred,thresh)
    target_boxes = region_layer.lastlayer2detection(target,thresh)
    
    
    #padding
    max_num_boxes = 20
    for num_batch in range(len(pred_boxes)):
        for i in range(max_num_boxes - len(pred_boxes[num_batch])):
            pred_boxes[num_batch].append([0.0,0.0,0.0,0.0,0.0,0.0])
        for i in range(max_num_boxes - len(target_boxes[num_batch])):
            target_boxes[num_batch].append([0.0,0.0,0.0,0.0,0.0,0.0])
    pred_boxes = np.array(pred_boxes)
    target_boxes = np.array(target_boxes)
    
    image_ids = np.array([str(uuid.uuid4()) for i in range(len(pred_boxes))])
    pred_clss = pred_boxes[...,0]
    pred_prob = pred_boxes[...,1]
    pred_boxes = np.array(pred_boxes[...,2:] * 200,dtype = np.int)
    target_clss = target_boxes[...,0]
    target_boxes = np.array(target_boxes[...,2:] * 200 ,dtype = np.int)

    
    return [image_ids,pred_clss,pred_prob,pred_boxes,target_clss,target_boxes]
    
    

    
    
    
    
    
    
    
    
  


