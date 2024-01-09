from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
from PIL.Image import Image
from torchvision.utils import save_image

from chinopie import logger
from chinopie.filehelper import InstanceFileHelper,GlobalFileHelper

from sd_hook import patch_pipe,tune_lora_scale

pipe:StableDiffusionPipeline = StableDiffusionPipeline.from_single_file('base_models/meinamix_meinaV11.safetensors',local_files_only=True).to('cuda')
pipe.safety_checker=None
logger.warning("loaded pipeline")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
logger.warning("loaded scheduler")

helper=GlobalFileHelper('deps')

ti_ckpt=torch.load(helper.get_exp_instance('arona(0)_trial1').find_latest_checkpoint())['model']
lora_unet_ckpt=torch.load(helper.get_exp_instance('arona(1)_trial1').find_latest_checkpoint())['model']
patch_pipe(pipe,ti_ckpt=ti_ckpt,unet_ckpt=lora_unet_ckpt)
tune_lora_scale(pipe.unet,0.75)

prompt = "a girl <arona1> <arona2> <arona3> <arona4>, blue hair, short skirt, white stocking"
for i in range(10):
    torch.manual_seed(i) # !
    image:Image = pipe(prompt, num_inference_steps=50, guidance_scale=7,num_images_per_prompt=1,clip_skip=2).images[0]

    image.save(f'test-{i}.jpg')