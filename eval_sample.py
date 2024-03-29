from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
from PIL.Image import Image
from torchvision.utils import save_image

from chinopie import logger
from chinopie.filehelper import InstanceFileHelper,GlobalFileHelper

from sd_hook import patch_pipe

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
logger.warning("loaded pipeline")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
logger.warning("loaded scheduler")

helper=GlobalFileHelper('deps')
for i in range(0,100,10):
    ti_ckpt=torch.load(helper.get_exp_instance('alpha').get_checkpoint_slot(i))['model']
    patch_pipe(pipe,ti_ckpt=ti_ckpt)

    prompt = "a girl in the style of <clear>"
    torch.manual_seed(0) # !
    image:Image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]

    image.save(f'test-{i}.jpg')