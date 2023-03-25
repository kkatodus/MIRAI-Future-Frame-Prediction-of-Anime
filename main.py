# %%
from vatt.experiments import pretrain
from data_proc.data_procviser import visualize_np_sequence_opencv, output_images_from_np_sequence
import numpy as np
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

#%%
#setting up the parameters
FRAME_SIZE = 128
NUM_FRAMES = int(20)
NUM_TRAINING_FRAMES = int(20/2)
NUM_TEST_FRAMES = int(20/2)
NUM_CHANNELS = 3

# %%
# loading data of numpy

#(BATCH, 20, 128, 128, 3)
video_data_np = np.load('./data/dataset/numpy/0.npy')
video_shape = video_data_np.shape
print("video_data_np", video_data_np.shape)
visualize_np_sequence_opencv(video_data_np[14], video_name="video.mp4", fps=15)


# %%
# need to create the configs of the model with config_factory
# following the steps from ./vatt/main.py
params = config_factory.build_experiment_configs(
    task='pretrain', model_arch='tx_fac')
params.eval.input.name = ['anime_ds']
# params.eval.input.raw_audio = True
params.eval.input.num_frames = NUM_TRAINING_FRAMES
params.eval.input.frame_size = FRAME_SIZE
params.model_config.backbone_config.video_backbone = 'vit_base'

params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_bbs_ckpt-500000",
    'model_dir': './results/',
    'mode': 'eval',
    'strategy_config': {'distribution_strategy': 'mirrored'},
})


# %%
# referencing ./vatt/main.py
executor = pretrain.get_executor(params=params)


# %%
# loading checkpoint
# following the steps from ./vatt/experiments/base.py

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
BATCH_NUM = 20
inputs, labels = executor.prepare_inputs({"vision":video_data_np[:BATCH_NUM, :NUM_TRAINING_FRAMES]})
outputs = executor.model(inputs, training = False)
print("outputs")
print(outputs.keys())
encoded_video_p = outputs['video']['features_pooled']
encoded_video = outputs['video']['features']
encoded_audio_p = outputs['audio']['features_pooled']
encoded_audio = outputs['audio']['features']
encoded_text_p = outputs['text']['features_pooled']
encoded_text = outputs['text']['features']
print("encoded_video", encoded_video.shape)
print("encoded_video_p", encoded_video_p.shape)
print("encoded_audio", encoded_audio.shape)
print("encoded_audio", encoded_audio_p.shape)
print("encoded_text", encoded_text.shape)
print("encoded_text_p", encoded_text_p.shape)


# %%
#decoder for the encoded_video 
#referencing training loop from https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
#https://keras.io/examples/vision/conv_lstm/
from tensorflow.keras import layers
from tensorflow import keras
VIDEO_ENCODING_SIZE = encoded_video.shape[1]
AUDIO_ENCODING_SIZE = encoded_audio.shape[1]
TEXT_ENCODING_SIZE = encoded_text.shape[1]

class ConvLSTM2DDecoder():
    def __init__(self,):
      super(ConvLSTM2DDecoder, self).__init__()
      self.input = keras.Input(shape=(int(NUM_TRAINING_FRAMES), int(FRAME_SIZE), int(FRAME_SIZE), int(NUM_CHANNELS)), batch_size = BATCH_NUM)
      self.conv_lstm = layers.ConvLSTM2D(6, kernel_size = (3,3), padding='same', return_sequences=True,stateful=True)

      lstm_out = self.conv_lstm(self.input)

      self.model = keras.Model(inputs=self.input, outputs=[lstm_out,])

    def __call__(self,input, initial_state=None):
       return self.model(input)



decoder = ConvLSTM2DDecoder()
decoder.model.summary()


# %%
reshaped_encoding = tf.reshape(encoded_video, (BATCH_NUM, 128, 128, 6))
model_input = video_data_np[:20, :NUM_TRAINING_FRAMES]
print("model_input", model_input.shape)
print("encoded_video", encoded_video.shape)
decoder.conv_lstm.reset_states(states = (reshaped_encoding, reshaped_encoding))
model_output = decoder(model_input)
print("model_output", model_output.shape)
   
# %%
for i in range(model_output.shape[0]):
  single_model_output = np.array(model_output[i, :, :, :, 1:4])
  single_model_output = (single_model_output - np.min(single_model_output))*255/(np.max(single_model_output) - np.min(single_model_output))
  single_model_output = single_model_output.astype(np.uint8)
  visualize_np_sequence_opencv(video_data_np[i], video_name=f"{i}_original.mp4", fps=15, dir='./results/')
  visualize_np_sequence_opencv(single_model_output, video_name=f"{i}_video.mp4", fps=15, dir='./results/')
# %%
