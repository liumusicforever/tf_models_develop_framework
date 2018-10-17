import tensorflow as tf



import networks.resnet50_DANet.resnet_v2 as resnet_v2
import networks.resnet50_DANet.params as config
import networks.resnet50_DANet.facenet as facenet


slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

def model_fn(features, labels, mode , params):
    """Model function for CNN."""
    
    # Subtracts the given means from each image channel.
    features.set_shape([None,224,224,3])
    num_channels = features.get_shape().as_list()[-1]
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=features)
    for i in range(num_channels):
        channels[i] = (channels[i] - (MEANS[i]/255.0))*255.0
    features = tf.concat(axis=3, values=channels)
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        phase_train_placeholder = True
    else:
        phase_train_placeholder = False
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          net, end_points = resnet_v2.resnet_v2_50(features,
                                                is_training=phase_train_placeholder,
                                                global_pool=True,
                                                spatial_squeeze = False)
    
    if config.pre_trained:
        exclude = []
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        
        tf.train.init_from_checkpoint(config.pre_trained, 
                          {v.name.split(':')[0]: v for v in variables_to_restore})
        
    '''
    Implementation Reference : 
        * Dual Attention Network for Scene Segmentation(https://arxiv.org/pdf/1809.02983.pdf)
        * https://github.com/junfu1115/DANet
    '''
    with tf.variable_scope("DANet"):
        danet_feat , pam_feat , cam_feat = DANet(net,is_training=phase_train_placeholder)
    
    
    total_danet_feat = tf.concat([danet_feat,pam_feat,cam_feat],axis = -1)
    net = slim.flatten(total_danet_feat)
    
    with slim.arg_scope([slim.batch_norm], is_training=phase_train_placeholder):
        prelogits = slim.fully_connected(net, config.embedding_size, 
                            scope='Bottleneck',
                            reuse=False)
        
    prelogits = slim.dropout(prelogits, keep_prob=0.5,
                                           is_training=phase_train_placeholder,
                                           scope='dropout')
    with slim.arg_scope([slim.batch_norm], is_training=phase_train_placeholder):
        logits = slim.fully_connected(prelogits, config.num_classes, activation_fn=None, 
                    weights_initializer=slim.initializers.xavier_initializer(), 
                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                    scope='Logits', reuse=False)
    
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    
    # Norm for the prelogits
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=config.prelogits_norm_p, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * config.prelogits_norm_loss_factor)

    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "embeddings" : embeddings
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



    # Add center loss
    prelogits_center_loss, _ = facenet.center_loss(prelogits, labels, config.center_loss_alfa, config.num_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * config.center_loss_factor)

#     global_step = tf.Variable(0,dtype = tf.float32)
    learning_rate = tf.train.exponential_decay(config.lr, tf.train.get_global_step(),
        config.learning_rate_decay_epochs*config.epoch_size, config.learning_rate_decay_factor, staircase=True)


    tf.summary.scalar('learning_rate', learning_rate)

    # Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

#         # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, tf.train.get_global_step(), config.optimizer, 
            config.lr, config.moving_average_decay, tf.global_variables(), config.log_histograms)
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : total_loss,"acc":accuracy}, every_n_iter=1)
        
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=total_loss, 
                                          train_op=train_op,
                                          training_hooks = [logging_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops,predictions = predictions)



def DANet(net,is_training = False):
    batch_size , h , w , c = net.get_shape().as_list()
    


    with slim.arg_scope([slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params={'is_training': is_training}):
        # net = slim.repeat(net, 2, slim.conv2d, c/4, [3, 3], scope='conv5')
        
        conv5p = slim.conv2d(net, c / 4 , [3, 3], scope='conv5p')
        conv5c = slim.conv2d(net, c / 4 , [3, 3], scope='conv5c')
    

    # PAM
    with tf.variable_scope("PAM"):
        net_dim = tf.shape(conv5p)
        # (batch_size , -1 , H x W)
        pam_feat_dims = tf.stack([net_dim[0], -1, net_dim[1] * net_dim[2]])

        pam_query = tf.reshape(
                            slim.conv2d(conv5p, c / 32 , [1, 1], scope='query'),
                            shape = pam_feat_dims)
        pam_key = tf.reshape(
                            slim.conv2d(conv5p, c / 32 , [1, 1], scope='key'),
                            shape = pam_feat_dims)
        pam_value = tf.reshape(
                            slim.conv2d(conv5p, c / 4 , [1, 1], scope='value'),
                            shape = pam_feat_dims)
        
        tf_pam_query = tf.transpose(pam_query, perm=[0, 2, 1])
        pam_attention = tf.nn.softmax(tf.matmul(tf_pam_query,pam_key),name="attention")
        tf_pam_attention = tf.transpose(pam_attention, perm=[0, 2, 1])
        pam_out = tf.matmul(pam_value,tf_pam_attention)
        pam_out = tf.reshape(pam_out,shape = net_dim)
        pam_gamma = tf.Variable(0.0, name="gamma")

        pam_out = pam_out * pam_gamma + conv5p

    # CAM
    with tf.variable_scope("CAM"):
        
        net_dim = tf.shape(conv5c)
        # (batch_size , -1 , H x W)
        cam_feat_dims = tf.stack([net_dim[0], net_dim[3] ,-1])
        cam_query = tf.reshape(conv5c,shape = cam_feat_dims)
        cam_key = tf.reshape(conv5c,shape = cam_feat_dims)
        cam_value = tf.reshape(conv5c,shape = cam_feat_dims)
        

        tf_cam_key= tf.transpose(cam_key, perm=[0, 2, 1])
        cam_energy = tf.matmul(cam_query,tf_cam_key,name="energy")
        cam_energy_new = tf.reduce_max(cam_energy,axis=-1,keepdims=True) - cam_energy
        cam_attention = tf.nn.softmax(cam_energy_new)
        
        cam_out = tf.matmul(cam_attention,cam_value)
        cam_out = tf.reshape(cam_out,shape = net_dim)
        cam_gamma = tf.Variable(0.0, name="gamma")

        cam_out = cam_out * cam_gamma + conv5c


    with slim.arg_scope([slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params={'is_training': is_training}):
        # net = slim.repeat(net, 2, slim.conv2d, c/4, [3, 3], scope='conv5')
        
        conv6p = slim.conv2d(pam_out, c / 4 , [3, 3], scope='conv6p')
        drop6p = slim.dropout(conv6p, keep_prob=0.9,
                                           is_training=is_training,
                                           scope='drop6p')
        conv7p = slim.conv2d(drop6p, c / 4 , [1, 1], scope='conv7p')



        conv6c = slim.conv2d(cam_out, c / 4 , [3, 3], scope='conv6c')
        drop6c = slim.dropout(conv6c, keep_prob=0.9,
                                           is_training=is_training,
                                           scope='drop6c')
        conv7c = slim.conv2d(drop6c, c / 4 , [1, 1], scope='conv7c')

        feat_sum = conv7p + conv7c
        conv8 = slim.conv2d(feat_sum, c / 4 , [1, 1], scope='conv8')

    return conv8 , conv7p ,conv7c
