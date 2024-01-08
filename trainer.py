import itertools
from typing import Any, Dict, List, Optional

import torch
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import Tensor
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset,DataLoader
from chinopie import chinopie, ModuleRecipe,TrainingRecipe, TrainBootstrap, logger
from chinopie.modelhelper import HyperparameterManager, ModelStaff
from chinopie.optim import LinearWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from transformers import CLIPTextModel,CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from models import get_models
from dataset import PivotalTuningDatasetCapation
import sd_hook

@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values).contiguous().float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    if cached_latents:
        train_dataloader = DataLoader(
            cached_latents_dataset, # type: ignore
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        print("PTI : Using cached latent.")
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader

class ModelWrapper(nn.Module):
    def __init__(self,text_encoder:CLIPTextModel,vae:AutoencoderKL,unet:UNet2DConditionModel):
        super().__init__()

        self.text_encoder=text_encoder
        self.vae=vae
        self.unet=unet


class BaseRecipe(TrainingRecipe):
    model:ModelWrapper

    def __init__(self, dataset:str):
        super().__init__(clamp_grad=1.0)

        self.dataset=dataset
    

    def ask_hyperparameter(self, hp: HyperparameterManager):
        self.pretrained_model_name = hp.suggest_category(
            "pretrained_model_name", ["runwayml/stable-diffusion-v1-5","meinamix"]
        )

        self.batch_size=hp.suggest_int('batch_size',1,16,log=True)        
        self.t_mutliplier=hp.suggest_float('t_mutliplier',0,1,log=True)
        self.template_type=hp.suggest_category('template_type',['object','style'])

        self.resolution=512
        self.color_jitter=True
        self.use_face_segmentation_condition=False
        self.use_mask_captioned_data=False
        self.train_inpainting=False
    
    def prepare(self, staff: ModelStaff):
        self.noise_scheduler=None
    
    def get_dataset(self,staff:ModelStaff,tokenizer:CLIPTokenizer,token_map:Dict[str,str]):
        assert type(self.template_type)==str, "value of wrong type is given to `template`"
        train_dataset = PivotalTuningDatasetCapation(
            instance_data_root=staff.file.get_dataset_slot(self.dataset),
            token_map=token_map,
            use_template=self.template_type,
            tokenizer=tokenizer,
            size=self.resolution,
            color_jitter=self.color_jitter,
            use_face_segmentation_condition=self.use_face_segmentation_condition,
            use_mask_captioned_data=self.use_mask_captioned_data,
            train_inpainting=self.train_inpainting,
        )
        train_dataset.blur_amount = 200 # I do not know what this does
        return train_dataset

    def forward(self, data) -> Any:
        assert self.noise_scheduler is not None, "noise scheduler must be init"

        # TODO: support cached latents
        latents=self.model.vae.encode(data['pixel_values'])['latent_dist'].sample() # type: ignore
        latents=latents*0.18215 # https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml#L17

        # TODO: support inpainting

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(
            0,
            int(self.noise_scheduler.config['num_train_timesteps'] * self.t_mutliplier),
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) # type: ignore
        latent_model_input = noisy_latents

        encoder_hidden_states = self.model.text_encoder(
            data["input_ids"]
        )[0]

        model_pred = self.model.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        return {
            'noise':noise,
            'timesteps':timesteps,
            'latents':latents,
            'pred':model_pred,
        }
    
    def cal_loss(self, data, output) -> Tensor:
        assert self.noise_scheduler is not None, "noise scheduler must be init"

        if self.noise_scheduler.config['prediction_type'] == "epsilon":
            target = output['noise']
        elif self.noise_scheduler.config['prediction_type'] == "v_prediction":
            target = self.noise_scheduler.get_velocity(output['latents'], output['noise'], output['timesteps'])
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config['prediction_type']}")
        
        # TODO: support mask

        loss=F.mse_loss(output['pred'],target.float(),reduction='none').mean([1,2,3]).mean()
        return loss

    def report_score(self, phase: str) -> float:
        return 0

class TextInversionRecipe(BaseRecipe):
    model:ModelWrapper
    def __init__(self,data_path:str,placeholder_tokens:List[str],init_tokens:List[str]):
        super().__init__(data_path)

        self.placeholder_tokens = placeholder_tokens
        self.init_tokens = init_tokens
    
    def ask_hyperparameter(self, hp: HyperparameterManager):
        super().ask_hyperparameter(hp)
        self.lr_ti=hp.suggest_float('lr_ti',0,1,log=True)
        self.weight_decay_ti=hp.suggest_float('weight_decay_ti',0,1,log=True)
        self.clip_ti_decay=hp.suggest_category('clip_ti_decay', [True, False])
        self.lr_text=hp.suggest_float('lr_text',0,1,log=True)
    

    def prepare(self, staff: ModelStaff):
        super().prepare(staff)

        assert type(self.pretrained_model_name)==str
        self.text_encoder,self.vae,self.unet,self.tokenizer,self.placeholder_token_ids,self.noise_scheduler = get_models(
            self.pretrained_model_name,
            pretrained_vae_name_or_path=None,
            revision="main",
            placeholder_tokens=self.placeholder_tokens,
            initializer_tokens=self.init_tokens,
            device='cpu',
        )

        # reg dataset
        token_map={'DUMMY':''.join(self.placeholder_tokens)}
        train_dataset=self.get_dataset(staff,self.tokenizer,token_map)
        train_loader=text2img_dataloader(train_dataset,self.batch_size,self.tokenizer,self.vae,self.text_encoder,cached_latents=False)
        staff.reg_dataset(train_dataset,train_loader,train_dataset,train_loader)

        chinopie.freeze_model(self.unet)
        chinopie.freeze_model(self.vae)
        params_to_freeze=itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            chinopie.freeze_model(param)
        
        staff.reg_model(ModelWrapper(self.text_encoder,self.vae,self.unet))
        # TODO: if cached latents, set vae to None

        # backup original embedding
        self.original_embeddings=self.text_encoder.get_input_embeddings().weight.data.clone() # type: ignore

        self.index_updates=torch.zeros(len(self.tokenizer)).bool()
        for id in self.placeholder_token_ids:
            self.index_updates[id]=True
        self.index_no_updates=~self.index_updates
        logger.debug(f"token inversion: {self.index_updates.int().sum()} tokens will be learned, leaving {self.index_no_updates.int().sum()} unchanged.")
    
    def export_model_state(self):
        learned_embeds_dict = {}
        for tok, tok_id in zip(self.placeholder_tokens, self.placeholder_token_ids):
            learned_embeds = self.model.text_encoder.get_input_embeddings().weight[tok_id] # type: ignore
            logger.debug(
                f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                learned_embeds[:4],
            )
            learned_embeds_dict[tok] = learned_embeds.detach().cpu()
        
        return learned_embeds_dict
    
    def import_model_state(self, state):
        sd_hook.apply_learned_embed_in_clip(state,self.model.text_encoder,self.tokenizer)

    def set_optimizers(self, model:ModelWrapper) -> Optimizer:
        return optim.AdamW(
            model.text_encoder.get_input_embeddings().parameters(),
            lr=self.lr_ti,
            betas=(0.9,0.999),
            eps=1e-8,
            weight_decay=self.weight_decay_ti
        )
    
    def set_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        return LinearWarmupScheduler(optimizer,num_warmup_steps=0,num_training_steps=self.total_epoch)
    
    def switch_train(self, model: ModelWrapper):
        super().switch_train(model)
        chinopie.set_eval(model.unet)
    
    def after_iter_train(self, data, output):
        with torch.no_grad():
            if self.clip_ti_decay:
                pre_norm = (
                    self.model.text_encoder.get_input_embeddings()
                    .weight[self.index_updates, :] # type: ignore
                    .norm(dim=-1, keepdim=True)
                )

                assert self.scheduler is not None
                lambda_ = min(1.0, 100 * self.scheduler.get_last_lr()[0])
                self.model.text_encoder.get_input_embeddings().weight[ # type: ignore
                    self.index_updates
                ] = F.normalize(
                    self.model.text_encoder.get_input_embeddings().weight[ # type: ignore
                        self.index_updates, :
                    ],
                    dim=-1,
                ) * (
                    pre_norm + lambda_ * (0.4 - pre_norm)
                )
                logger.debug(pre_norm)
            
            current_norm = (
                self.model.text_encoder.get_input_embeddings()
                .weight[self.index_updates, :] # type: ignore
                .norm(dim=-1)
            )

            self.model.text_encoder.get_input_embeddings().weight[ # type: ignore
                self.index_no_updates
            ] = self.original_embeddings[self.index_no_updates].to(self.dev)
            logger.debug(f"Current Norm:\n{current_norm}")
    
    def export_custom_state(self):
        return {
            'type':'text inversion'
        }


class LoRARecipe(BaseRecipe):
    def __init__(self, dataset: str):
        super().__init__(dataset)

    def ask_hyperparameter(self, hp: HyperparameterManager):
        super().ask_hyperparameter(hp)

        self.joint_optimization=hp.suggest_category('joint_optimization',[False,True])
        self.lr_unet=hp.suggest_float('lr_unet',0,1,log=True)
        self.lora_rank=hp.suggest_int('lora_rank',1,512,log=True)
        self.weight_decay_lora=hp.suggest_float('weight_decay_lora',0,1,log=True)
        self.dropout_lora=hp.suggest_float('dropout_lora',0,1,log=True)
    
    def prepare(self, staff: ModelStaff):
        super().prepare(staff)

        assert type(self.pretrained_model_name)==str
        self.text_encoder,self.vae,self.unet,self.tokenizer,self.placeholder_token_ids,self.noise_scheduler = get_models(
            self.pretrained_model_name,
            pretrained_vae_name_or_path=None,
            revision="main",
            placeholder_tokens=[],
            initializer_tokens=[],
            device='cpu',
        )
        
        if staff.prev_files is not None:
            logger.warning('found previous recipe. trying merging...')
            ckpt_path=staff.prev_files[-1].find_latest_checkpoint()
            assert ckpt_path is not None
            ckpt=torch.load(ckpt_path,map_location='cpu')
            assert ckpt['custom']['type']=='text inversion'
            logger.info(f"previous recipe is text inversion")
            placeholder_tokens=sd_hook.apply_learned_embed_in_clip(ckpt['model'],self.text_encoder,self.tokenizer)
            logger.info(f"injected new tokens: {placeholder_tokens}")
            
            assert isinstance(placeholder_tokens,List)
            token_map={'DUMMY':''.join(placeholder_tokens)}
        else:
            raise NotImplementedError('unknown token map')
        
        # reg dataset
        train_dataset=self.get_dataset(staff,self.tokenizer,token_map)
        train_loader=text2img_dataloader(train_dataset,self.batch_size,self.tokenizer,self.vae,self.text_encoder,cached_latents=False)
        staff.reg_dataset(train_dataset,train_loader,train_dataset,train_loader)
        
        # freeze model
        chinopie.freeze_model(self.unet)
        chinopie.freeze_model(self.vae)
        if not self.joint_optimization:
            chinopie.freeze_model(self.text_encoder)
            logger.info('frozen text')
        else:
            # TODO: allow tuning text
            logger.info('enabled text inversion optimization')
            raise NotImplementedError()

        # inject lora and collect trainable params
        unet_lora_params,_=sd_hook.inject_trainable_lora(self.unet,r=self.lora_rank,dropout_p=self.dropout_lora) # TODO: allow change more params
        logger.info(f"injected {len(unet_lora_params)} lora layers")
        self.params_to_optimize=[
            {"params": itertools.chain(*unet_lora_params), "lr": self.lr_unet},
        ]

        staff.reg_model(ModelWrapper(self.text_encoder,self.vae,self.unet))
    
    def set_optimizers(self, model) -> Optimizer:
        return optim.AdamW(
            self.params_to_optimize,lr=self.lr_unet,weight_decay=self.weight_decay_lora
        )
    
    def set_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        return LinearWarmupScheduler(optimizer,num_warmup_steps=0,num_training_steps=self.total_epoch)
    
    def export_model_state(self):
        return sd_hook.export_lora_weight(self.model.unet)
    
    def import_model_state(self, state):
        sd_hook.monkeypatch_or_replace_lora(
            self.model.unet,
            state,
            r=self.lora_rank,
        )
    

if __name__ == "__main__":
    tb = TrainBootstrap(
        "deps",
        num_epoch=300,
        load_checkpoint=True,
        save_checkpoint=True,
        checkpoint_save_period=10,
        enable_prune=True,
        seed=721,
        diagnose=False,
        verbose=False,
        dev='cuda',
        comment='alpha',
    )

    dataset = chinopie.get_env("dataset")
    placeholder_tokens = list(
        map(lambda x: x.strip(), chinopie.get_env("placeholder_tokens").split(","))
    )
    init_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    logger.warning(f"token:\nplaceholders: {placeholder_tokens}\ninit with: {init_tokens}")
    tb.hp.reg_category("pretrained_model_name",'meinamix')
    tb.hp.reg_int("batch_size", 1)
    tb.hp.reg_float('t_mutliplier',1.0)
    tb.hp.reg_float('lr_text',1e-5)
    tb.hp.reg_float('lr_ti',5e-4)
    tb.hp.reg_float('weight_decay_ti',0.0)
    tb.hp.reg_category('clip_ti_decay', True)
    tb.hp.reg_category('template_type','style')
    tb.hp.reg_float('lora_rank',4)
    tb.hp.reg_float('lr_unet',1e-4)
    tb.hp.reg_float('weight_decay_lora',1e-3)
    tb.hp.reg_float('dropout_lora',0)
    tb.hp.reg_category('joint_optimization',False)

    tb.optimize(
        TextInversionRecipe(dataset,placeholder_tokens, init_tokens),
        direction="maximize",
        inf_score=-1,
        n_trials=1,
        stage=0,
    )
    tb.optimize(
        LoRARecipe(dataset),
        direction="maximize",
        inf_score=-1,
        n_trials=1,
        stage=1,
    )

    tb.release()
