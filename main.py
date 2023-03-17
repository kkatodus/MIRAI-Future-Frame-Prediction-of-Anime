# %%
from vatt.modeling.factory import build_model
from vatt.configs import factory as config_factory
import tensorflow as tf

# %%
#need to create the configs of the model with config_factory
#following the steps from ./vatt/main.py
params = config_factory.build_experiment_configs(task = 'finetune', model_arch = 'tx_fac')
params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_bbs_ckpt-500000.data-00000-of-00001", 
    'model_dir':'./results/'
    
})
#initializing model
model_config = params.model_config
model = build_model(params = model_config)

# %%
from vatt.experiments.base import get_optimizer_step
#loading checkpoint
#following the steps from ./vatt/experiments/base.py
model_dir = params.model_dir
checkpoint_path = params.checkpoint_path
checkpoint = tf.train.Checkpoint(model = model)
checkpoint.restore(checkpoint_path).expect_partial().assert_existing_objects_matched()
current_step = get_optimizer_step(checkpoint)


# %%
