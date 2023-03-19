# %%
from vatt.modeling.factory import build_model
from vatt.configs import factory as config_factory
import tensorflow as tf

# %%
from vatt.data import processing
from vatt.modeling.backbones.video import factory as vid_factory

#need to create the configs of the model with config_factory
#following the steps from ./vatt/main.py
params = config_factory.build_experiment_configs(task = 'finetune', model_arch = 'tx_fac')
params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_bbs_ckpt-500000.data-00000-of-00001", 
    'model_dir':'./results/'
    
})
#referencing how they do it in ./vatt/experiments/finetune.py line 178 - 
input_params = params.train.input
space_to_depth = input_params.space_to_depth
input_shape = processing.get_video_shape(input_params, is_space_to_depth = space_to_depth)
inputs ={"images": tf.keras.Input(shape = input_shape, name = 'input')}
#initializing model
model_config = params.model_config
model = vid_factory.build_model(params = params.model_config, mode='predict')
outputs = model(inputs, None)
keras_model = tf.keras.Model(inputs = inputs, outputs = outputs)
keras_model.loss_fn = model.loss_fn

# %%
from vatt.experiments.base import get_optimizer_step
#loading checkpoint
#following the steps from ./vatt/experiments/base.py
model_dir = params.model_dir
checkpoint_path = params.checkpoint_path
checkpoint = tf.train.Checkpoint(model = keras_model)
checkpoint.restore(checkpoint_path).expect_partial().assert_existing_objects_matched()
current_step = get_optimizer_step(checkpoint)


# %%
