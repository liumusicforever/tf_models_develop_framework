import tensorflow as tf


import networks.resnet50_selfatten.resnet_v2 as resnet_v2
import networks.resnet50_selfatten.params as config


slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def model_fn(features, labels, mode, params):
    """Model function for CNN."""

    # Subtracts the given means from each image channel.
    features.set_shape([None, 224, 224, 3])
    num_channels = features.get_shape().as_list()[-1]
    channels = tf.split(
        axis=3, num_or_size_splits=num_channels, value=features)
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
                                                 spatial_squeeze=False)
    if config.pre_trained:
        exclude = ['global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(
            exclude=exclude)

        tf.train.init_from_checkpoint(config.pre_trained,
                                      {v.name.split(':')[0]: v for v in variables_to_restore})
    net = slim.flatten(net)

    '''End of resnet50 backbone feature extraction'''

    logits = classifier(net, phase_train_placeholder)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "embeddings": net
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        correct_prediction = tf.cast(
            tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        total_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        learning_rate = tf.train.exponential_decay(config.lr, tf.train.get_global_step(),
                                                   config.learning_rate_decay_epochs*config.epoch_size, config.learning_rate_decay_factor, staircase=True)

        opt_backbone = tf.train.MomentumOptimizer(
            config.lr * 0.1, 0.9, use_nesterov=True)
        opt_lastlayer = tf.train.MomentumOptimizer(
            config.lr, 0.9, use_nesterov=True)
        backbone_vars = [
            i for i in tf.trainable_variables() if 'resnet_v2_50/' in i.name]
        last_vars = [
            i for i in tf.trainable_variables() if 'resnet_v2_50/' not in i.name]

        backbone_grads = opt_backbone.compute_gradients(
            total_loss, backbone_vars)
        lastlayer_grads = opt_lastlayer.compute_gradients(
            total_loss, last_vars)

        backbone_grads = opt_backbone.compute_gradients(
            total_loss, backbone_vars)
        lastlayer_grads = opt_lastlayer.compute_gradients(
            total_loss, last_vars)
        # Apply gradients.
        op_backbone = opt_backbone.apply_gradients(
            backbone_grads, global_step=tf.train.get_global_step())
        op_lastlayer = opt_lastlayer.apply_gradients(
            lastlayer_grads, global_step=tf.train.get_global_step())
        apply_gradient_op = tf.group(op_backbone, op_lastlayer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        logging_hook = tf.train.LoggingTensorHook(
            {"loss": total_loss, "acc": accuracy}, every_n_iter=1)
        with tf.control_dependencies([apply_gradient_op]+update_ops):
            train_op = tf.no_op(name='train')
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook])
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops, predictions=predictions)


def classifier(net, is_training):
    with slim.arg_scope([slim.batch_norm],
                        is_training=is_training,
                        epsilon=1.0,
                        renorm_decay=0.99):
        net = slim.fully_connected(net,
                                   2048,
                                   activation_fn=tf.nn.leaky_relu,
                                   weights_initializer=slim.initializers.xavier_initializer(),
                                   scope='Bottleneck',
                                   reuse=False,
                                   normalizer_fn=slim.batch_norm
                                   )
        net = slim.dropout(net,
                           keep_prob=0.5,
                           is_training=is_training,
                           scope='dropout')
    with slim.arg_scope([slim.fully_connected]):
        net = slim.fully_connected(net,
                                   config.num_classes,
                                   weights_initializer=tf.truncated_normal_initializer(
                                       stddev=0.01),
                                   scope='Logits', reuse=False)
    return net
