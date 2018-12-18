batch_size = 75
save_summary_steps = 50
num_epochs = 25

network_params = {

}

pre_trained = '/root/dennis_code_base/tf_models_develop_framework/experiments/facenet_resnet50_pretrained/resnet_v2_50.ckpt'
lr = 0.1
keep_probability = 1.0
embedding_size = 2048
weight_decay = 0.0
# num_classes = 702
num_classes = 752
prelogits_norm_p = 1.0
prelogits_norm_loss_factor = 5e-4
center_loss_alfa = 0.5
center_loss_factor = 0.1
learning_rate_decay_epochs = 100
epoch_size = 1000
learning_rate_decay_factor = 1.0
optimizer = "MOM"
moving_average_decay = 0.9999
log_histograms = True