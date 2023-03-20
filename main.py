# %%
import tensorflow as tf
from vatt.configs import factory as config_factory
print('gpus',tf.config.list_physical_devices('GPU'))

#need to create the configs of the model with config_factory
#following the steps from ./vatt/main.py
params = config_factory.build_experiment_configs(task = 'pretrain', model_arch = 'tx_fac')
params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_mbs_ckpt-500000", 
    'model_dir':'./results/',
    'mode':'eval',
    'strategy_config':{'distribution_strategy': 'mirrored'},
    'train':{"input":{"name":""}},
    'eval':{'input':{"name":''}}
})

#%%
from vatt.experiments import pretrain
#referencing ./vatt/main.py
executor = pretrain.get_executor(params = params)

# %%
from vatt.experiments.base import get_optimizer_step
#loading checkpoint
#following the steps from ./vatt/experiments/base.py
keras_model = executor.model
optimizer = keras_model.optimizer
keras_model.summary()
checkpoint = tf.train.Checkpoint(model = executor.model, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory='./vatt/checkpoint', max_to_keep=1)
try:
    checkpoint.restore(params.checkpoint_path)
    print("Checkpoint restored successfully")
except:
    print("Loading the checkpoint failed")
# status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
# checkpoint_manager.save()
# checkpoint = tf.train.Checkpoint(model = keras_model, optimizer = optimizer)
# checkpoint.restore(checkpoint_path).expect_partial().assert_existing_objects_matched()


# %%

# %%
