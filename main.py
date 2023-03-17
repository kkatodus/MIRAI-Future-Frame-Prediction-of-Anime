# %%
from vatt.modeling.factory import build_model
from vatt.configs import factory as config_factory

# %%
#need to create the configs of the model with config_factory
params = config_factory.build_experiment_configs(task = 'finetune', model_arch = 'tx_fac')
params.override({
    'checkpoint_path': "./vatt/checkpoint/vision_vatt_pretrain_tx_fac_bbs_ckpt-500000.data-00000-of-00001", 
    
})
model_config = params.model_config
model = build_model(params = model_config)

# %%
print(model)

# %%
