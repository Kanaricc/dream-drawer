from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
from PIL.Image import Image
from torchvision.utils import save_image

from chinopie import logger
from chinopie.filehelper import InstanceFileHelper,GlobalFileHelper

from sd_hook import patch_pipe

pipe = StableDiffusionPipeline.from_single_file('base_models/meinamix_meinaV11.safetensors').to('cuda')
pipe.safety_checker=None
logger.warning("loaded pipeline")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
logger.warning("loaded scheduler")

helper=GlobalFileHelper('deps')

for i in range(0,300,10):
    ti_ckpt=torch.load(helper.get_exp_instance('alpha(0)').find_latest_checkpoint())['model']
    lora_unet_ckpt=torch.load(helper.get_exp_instance('alpha(1)').get_checkpoint_slot(i))['model']
    patch_pipe(pipe,ti_ckpt=ti_ckpt,unet_ckpt=lora_unet_ckpt)

    prompt = "a girl in the style of <clear>"
    torch.manual_seed(0) # !
    image:Image = pipe(prompt, num_inference_steps=40, guidance_scale=7,num_images_per_prompt=1).images[0]

    image.save(f'test/test-{i}.jpg')