from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
from PIL.Image import Image
from torchvision.utils import save_image

from chinopie import logger
from chinopie.filehelper import InstanceFileHelper,GlobalFileHelper

from sd_hook import patch_pipe

helper=GlobalFileHelper('deps')


pipe = StableDiffusionPipeline.from_single_file('base_models/meinamix_meinaV11.safetensors', torch_dtype=torch.float16)
pipe.safety_checker=None
import pdb
pdb.set_trace()