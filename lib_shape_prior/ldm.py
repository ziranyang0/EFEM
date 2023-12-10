from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
from copy import deepcopy
import wandb
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, mode="concat", normalize_level=0, codebook=None):
        assert mode in ["concat", "seperate"]
        self.mode = mode
        if codebook == None:
            codebook_path = "/home/ziran/se3/EFEM/cache/mugs.npz"
            with np.load(codebook_path) as data:
                codebook = {key: data[key] for key in data}
            del codebook['id']
            for k, v in codebook.items():
                if isinstance(v, np.ndarray):
                    newv = torch.from_numpy(v)
                    codebook[k] = newv
                print(k, v.shape)

        self.normalize_params = {
            "normalize_level": 0,
            'z_so3_mean': 0,
            'z_so3_std': 1,
            'z_inv_mean': 0,
            'z_inv_std': 1,
        }
        if normalize_level == 1:
            z_so3_mean = codebook['z_so3'].mean()
            z_so3_std = codebook['z_so3'].std()
            codebook['z_so3'] = (codebook['z_so3'] - z_so3_mean) / z_so3_std
            z_inv_mean = codebook['z_inv'].mean()
            z_inv_std = codebook['z_inv'].std()
            codebook['z_inv'] = (codebook['z_inv'] - z_inv_mean) / z_inv_std
            normalize_params = {
                "normalize_level": normalize_level,
                'z_so3_mean': z_so3_mean,
                'z_so3_std': z_so3_std,
                'z_inv_mean': z_inv_mean,
                'z_inv_std': z_inv_std,
            }
            print("normalized!")
            print(normalize_params)
            self.normalize_params = normalize_params
            
        if mode == "concate":
            # feat = torch.cat([codebook['z_so3'], #[B, 256, 3]
            #           codebook['z_inv'].unsqueeze(2).repeat(1,1,3), # [B,256] -> [B, 256, 3]
            #           codebook['scale'].unsqueeze(1).unsqueeze(1).repeat(1,1,3), # [B]-> [B,1,3]
            #           codebook['center']], dim=1) # [B, 1, 3]
            feat = torch.cat([codebook['z_so3'], #[B, 256, 3]
                    codebook['z_inv'].unsqueeze(2).repeat(1,1,3)], dim=1) # [B,256] -> [B, 256, 3]
                    #   codebook['scale'].unsqueeze(1).unsqueeze(1).repeat(1,1,3), # [B]-> [B,1,3]
                    #   codebook['center'].transpose(1,2).repeat(1,1,3)], dim=1) # [B, 3, 3]
                    # [B, 1, 3] -> [x,y,z]
            # !!! 注意这里乘了10
            feat = feat*10
            assert feat.shape[1:] == (512,3)
            self.feat = [feat[i] for i in range(feat.shape[0])]
            self.pcl = [codebook['pcl'][i] for i in range(codebook['pcl'].shape[0])]
            
        elif mode == "seperate":
            self.x = [codebook['z_so3'][i] for i in range(codebook['z_so3'].shape[0])]
            self.s = [codebook['z_inv'][i] for i in range(codebook['z_inv'].shape[0])]
            self.pcl = [codebook['pcl'][i] for i in range(codebook['pcl'].shape[0])]
            
            # self.x = self.x + self.x
            # self.s = self.s + self.s
            # self.pcl = self.pcl + self.pcl

        print("dataset loaded")

    def __len__(self):
        if self.mode == "concat":
            return len(self.feat)
        elif self.mode == "seperate":
            return len(self.x)

    def __getitem__(self, idx):
        if self.mode == "concat":
            feature = self.feat[idx]
            pcl_item = self.pcl[idx]
            return feature, pcl_item
        elif self.mode == "seperate":
            x = self.x[idx]
            s = self.s[idx]
            pcl_item = self.pcl[idx]
            return torch.concat([x,s.unsqueeze(1)], dim=1), pcl_item         

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 200000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1000
    save_model_epochs = 10000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "./dev_ckpt/12-9/"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0




import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(args, config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, train_ds):
    # Initialize accelerator and tensorboard logging
    report_to = "wandb" if not args.debug else None
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=report_to,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    log_loss = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # features = batch[0] # [B, 512, 3]
            features = batch[0] # [B, 256, 4]
            # clean_images = features # [B, 512, 3]
            clean_images = features # [B, 256, 3]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            
            # if epoch==1000:
            #     print("hello")
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                noise_pred_x, noise_pred_s  = model(noisy_images[:,:,:3], noisy_images[:,:,3], timesteps)
                noise_pred = torch.concat([noise_pred_x, noise_pred_s.unsqueeze(2)], dim=2)

                loss = F.mse_loss(noise_pred, noise)
                log_loss = loss.detach().item()
                accelerator.backward(loss)

                # accelerator.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            # progress_bar.set_postfix(**logs)
            print(logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # from diffusers import DDPMPipeline
            # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                save(model, log_loss, epoch, config.output_dir, train_ds)

import argparse

def save(model, loss, epoch, output_dir, train_ds):
    # if not os.path.exists('./ckpts'):
    #     os.mkdir('./ckpts')
    filename = f'{output_dir}/model_{epoch}.pt'
    obj = {
        'model': model.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'normalize_params': train_ds.normalize_params,
    }
    torch.save(obj, filename)

from dev_utils import ema_update, get_alphas_sigmas, get_ddpm_schedule, FourierFeatures, save_model

from vec_layers_out import VecLinearNormalizeActivate as VecLNA
from vec_layers_out import VecLinear
class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, scalar_hidden_dims, max_freq, num_bands, std=0.2):
        super(LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim
        self.scalar_hidden_dims = scalar_hidden_dims
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2*num_bands
        
        leak_neg_slope=0.2
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        self.layers = nn.ModuleList()
        self.skip_connections = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(VecLNA(in_features=self.latent_dim+self.t_emb_dim,
                                          out_features=hidden_dims[0],
                                          s_in_features=self.latent_dim+self.t_emb_dim,
                                          s_out_features=scalar_hidden_dims[0],
                                          mode="so3", 
                                        #   s_normalization=torch.nn.BatchNorm1d(scalar_hidden_dims[0]),
                                          act_func=act_func))
                if self.latent_dim + self.t_emb_dim == hidden_dims[0] and self.latent_dim + self.t_emb_dim == scalar_hidden_dims[0]:
                    self.skip_connections.append(True)
                else:
                    self.skip_connections.append(False)
            else:
                self.layers.append(VecLNA(in_features=hidden_dims[i-1],
                                          out_features=hidden_dims[i],
                                          s_in_features=scalar_hidden_dims[i-1],
                                          s_out_features=scalar_hidden_dims[i],
                                        #   s_normalization=torch.nn.BatchNorm1d(scalar_hidden_dims[i]),
                                          mode="so3", act_func=act_func))
                if hidden_dims[i-1] == hidden_dims[i] and scalar_hidden_dims[i-1] == scalar_hidden_dims[i]:
                    self.skip_connections.append(True)
                else:
                    self.skip_connections.append(False)
        
        self.vn_head = VecLinear(v_in=hidden_dims[-1],
                                 v_out=self.latent_dim,
                                 s_in=scalar_hidden_dims[-1],
                                 s_out=self.latent_dim,
                                 mode="so3",)
        # self.scalar_head = nn.Linear(scalar_hidden_dims[-1], self.latent_dim)
    
    def forward(self, z_so3, z_inv, t):
        t = t/1000
        batch_size = z_so3.size(0)
        # Embed time
        t_emb_s = self.timestep_embed(t)
        t_emb_x = self.timestep_embed(t).unsqueeze(2).repeat(1, 1, 3)
        z_inv = torch.cat([z_inv, t_emb_s], dim=1)
        z_so3 = torch.cat([z_so3, t_emb_x], dim=1)

        for i, layer in enumerate(self.layers):
            z_so3_new, z_inv_new = layer(z_so3, z_inv)
            if self.skip_connections[i]:
                z_so3 = z_so3_new + z_so3
                z_inv = z_inv_new + z_inv
            else:
                z_so3 = z_so3_new
                z_inv = z_inv_new
        
        # pred_z_so3 = self.vn_head(z_so3)
        # pred_z_inv = self.scalar_head(z_inv)
        pred_z_so3, pred_z_inv = self.vn_head(z_so3, z_inv)
        
        return pred_z_so3, pred_z_inv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        print("Debug mode is enabled")
    
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    config = TrainingConfig()
    seed = 0
    torch.manual_seed(seed)


    train_ds = CustomDataset(mode="seperate", normalize_level=0)
    train_dl = DataLoader(train_ds, batch_size=149, shuffle=True)
    
    # Create the model and optimizer
    latent_dim = 256

    # hidden_dims = [2048,4096, 4096, 2048]  # Example of hidden dimensions
    # hidden_dims = [4096,8192, 8192, 4096]  # Example of hidden dimensions
    hidden_dims = [2048, 2048, 2048, 2048] 
    # hidden_dims = [4096, 4096, 4096, 4096] 
    max_freq = 4  # Example max frequency for Fourier features
    num_bands = 4  # Number of frequency bands

    # 初始化模型
    # model = Diffusion().to(device)
    # diffusion_model = LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    # diffusion_model = Residual_LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    # diffusion_model = Residual_LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    scalar_hidden_dims = [256,256,256,256]
    model = LatentDiffusionModel(latent_dim, hidden_dims, scalar_hidden_dims, max_freq, num_bands).to(device)
    
    args.run_name = "residual_cos_200k_norm0_timeNorm"
    config.output_dir += args.run_name
    if not args.debug:
        wandb.init(project="vnDiffusion", entity="_zrrr", name=args.run_name)

        print('Diffusion Model parameters:', sum(p.numel() for p in model.parameters()))
        wandb.log({"max_freq": max_freq, "num_bands": num_bands,
                #    "device": device,
                "model_params": sum(p.numel() for p in model.parameters())})
    
    
    # model = LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    # # wandb.init(project="vnDiffusion", entity="_zrrr", name="run_vnVanilla_1024_2048_2048_1024")
    # if not args.debug:
    #     wandb.init(project="vnDiffusion", entity="_zrrr", name="cluster_ldm_vnVanilla_4096_8192_600k")
    #     # wandb.init(project="vnDiffusion", entity="_zrrr", name="cluster_ldm_vnVanilla_2048_4096_200k")

    #     print('Diffusion Model parameters:', sum(p.numel() for p in model.parameters()))
    #     wandb.log({"max_freq": max_freq, "num_bands": num_bands,
    #             #    "device": device,
    #             "model_params": sum(p.numel() for p in model.parameters())})
    



    from diffusers.optimization import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dl) * config.num_epochs),
    )


    train_loop(args, config, model, noise_scheduler, optimizer, train_dl, lr_scheduler, train_ds)



if __name__ == "__main__":
    main()