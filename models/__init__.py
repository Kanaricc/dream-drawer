import re
from typing import List,Optional

import torch
from torch import Tensor
from transformers import CLIPTextModel,CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from chinopie import logger

def _get_standard_models(pretrained_model_name_or_path:str,pretrained_vae_name_or_path:Optional[str],revision:str="main"):
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    assert isinstance(tokenizer,CLIPTokenizer)

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    assert isinstance(text_encoder,CLIPTextModel)

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    assert isinstance(vae,AutoencoderKL)
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )
    assert isinstance(unet,UNet2DConditionModel)

    return tokenizer,text_encoder,vae,unet

def _get_models_from_safetensors(path):
    pipe = StableDiffusionPipeline.from_single_file(path)
    assert isinstance(pipe,StableDiffusionPipeline)
    return pipe.tokenizer,pipe.text_encoder,pipe.vae,pipe.unet,pipe.scheduler


def get_models(
    pretrained_model_name_or_path:str,
    pretrained_vae_name_or_path:Optional[str],
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device,
    revision:str='main',
):
    if pretrained_model_name_or_path=='runwayml/stable-diffusion-v1-5':
        tokenizer,text_encoder,vae,unet=_get_standard_models(pretrained_model_name_or_path,pretrained_vae_name_or_path)
        default_scheduler=DDPMScheduler.from_config(pretrained_model_name_or_path=pretrained_model_name_or_path,subfolder='scheduler')
    elif pretrained_model_name_or_path in ['meinamix']:
        tokenizer,text_encoder,vae,unet,default_scheduler=_get_models_from_safetensors(f"base_models/meinamix_meinaV11.safetensors")
        logger.info(f"loaded custom base model `{pretrained_model_name_or_path}`")
    else:
        raise NotImplementedError(f"unknown model `{pretrained_model_name_or_path}`")


    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        assert isinstance(token_embeds, Tensor)
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            logger.info(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            logger.info(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
        default_scheduler,
    )