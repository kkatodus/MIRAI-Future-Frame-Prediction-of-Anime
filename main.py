# %%
import tensorflow as tf
from vatt.configs import factory as config_factory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# %%
#loading data of numpy
import numpy as np

video_data_np = np.load('./data/dataset/out_of_sight_1.npy')
video_shape = video_data_np.shape
print("video_shape_before", video_shape)
video_data_np = video_data_np[:30,100:200,100:200,:]
print("video_shape_after", video_data_np.shape)
video_data_np = np.expand_dims(video_data_np, axis = 0)
print("video_shape_after_expand", video_data_np.shape)


# %%
#need to create the configs of the model with config_factory
#following the steps from ./vatt/main.py
params = config_factory.build_experiment_configs(task = 'pretrain', model_arch = 'tx_fac')
params.eval.input.name = ['anime_ds']
# params.eval.input.raw_audio = True
params.eval.input.num_frames = 20
params.eval.input.frame_size = 64
params.model_config.backbone_config.video_backbone = 'vit_base'

params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_bbs_ckpt-500000", 
    'model_dir':'./results/',
    'mode':'eval',
    'strategy_config':{'distribution_strategy': 'mirrored'},
})


#%%
from vatt.experiments import pretrain
#referencing ./vatt/main.py
executor = pretrain.get_executor(params = params)


# %%
#loading checkpoint
#following the steps from ./vatt/experiments/base.py

checkpoint = tf.train.Checkpoint(model = executor.model, optimizer = executor.model.optimizer)
try:
    checkpoint.restore(params.checkpoint_path)
    print("Checkpoint restored successfully")
except:
    print("Loading the checkpoint failed")


# %%
#referencing how they do it in ./vatt/experiments/base.py line 253 - 
#prepare_inputs() in ./vatt/pretrain.py
#TODO: later add "audio" to the dict of input

inputs, labels = executor.prepare_inputs({"vision":video_data_np})
outputs = executor.model(inputs, training = False)
print("outputs")
print(outputs.keys())
encoded_video = outputs['video']['features_pooled']
encoded_audio = outputs['audio']['features_pooled']
encoded_text = outputs['text']['features_pooled']
print("encoded_video", encoded_video.shape)
print("encoded_audio", encoded_audio.shape)
print("encoded_text", encoded_text.shape)



# %%
