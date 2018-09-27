import tensorflow as tf



import networks.facenet_inception_resnet_v1.models.inception_resnet_v1 as base_models
import networks.facenet_inception_resnet_v1.params as config
import networks.facenet_inception_resnet_v1.facenet as facenet


slim = tf.contrib.slim

def model_fn(features, labels, mode , params):
    """Model function for CNN."""
    
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        phase_train_placeholder = True
    else:
        phase_train_placeholder = False
    
    
    # Build the inference graph
    prelogits, _ = base_models.inference(features, config.keep_probability, 
        phase_train=phase_train_placeholder, bottleneck_layer_size=config.embedding_size, 
        weight_decay=config.weight_decay)
    
    if config.pre_trained:
        
        exclude = ['global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        
        tf.train.init_from_checkpoint(config.pre_trained, 
                          {v.name.split(':')[0]: v for v in variables_to_restore})
    
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
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



    # Add center loss
    prelogits_center_loss, _ = facenet.center_loss(prelogits, labels, config.center_loss_alfa, config.num_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * config.center_loss_factor)

    global_step = tf.Variable(0,dtype = tf.float32)
    learning_rate = tf.train.exponential_decay(config.lr, global_step,
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
        
#         train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

#         # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, tf.train.get_global_step(), config.optimizer, 
            config.lr, config.moving_average_decay, tf.global_variables(), config.log_histograms)
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : total_loss}, every_n_iter=1)
        
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

