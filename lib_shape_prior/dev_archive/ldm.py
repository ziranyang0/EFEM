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
            
            self.x = self.x + self.x
            self.s = self.s + self.s
            self.pcl = self.pcl + self.pcl

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
    num_epochs = 1500000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1000
    save_model_epochs = 100000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "./dev_ckpt/12-2/"  # the model name locally and on the HF Hub

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

                noise_pred = model(noisy_images, timesteps)

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

class FourierFeatures(nn.Module):
    def __init__(self, max_freq, num_bands, std=1.0):
        super().__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.std = std

    def forward(self, t):
        freqs = torch.exp2(torch.linspace(0, self.max_freq, self.num_bands, device=t.device))
        steps = t[:, None] * freqs[None]
        embedding = torch.cat((steps.sin(), steps.cos()), dim=-1) * self.std
        return embedding

class origin_LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, max_freq, num_bands, std=0.2):
        super(origin_LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim*4
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2*num_bands
        
        self.layers = nn.ModuleList()
        leak_neg_slope=0.2
        self.act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(nn.Linear(in_features=self.latent_dim+self.t_emb_dim, 
                                             out_features=hidden_dims[0]))
            else:
                self.layers.append(nn.Linear(in_features=hidden_dims[i-1],
                                             out_features=hidden_dims[i]))

        # 最后一层输出与原始维度相同
        self.layers.append(nn.Linear(in_features=hidden_dims[-1],
                                     out_features=self.latent_dim))
        

    def forward(self, z_so3_and_z_inv, t):
        # z_so3_and_z_inv: [B, 256, 4]
        batch_size = z_so3_and_z_inv.size(0)
        # Embed time
        t_emb = self.timestep_embed(t)
        z_so3_and_z_inv = z_so3_and_z_inv.reshape(batch_size, -1) # [b, 256, 4] -> [b, 1024]
        feat = torch.cat([z_so3_and_z_inv, t_emb], dim=1) # [b, 1024+xxx]

        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                feat = layer(feat)
            else:
                feat = self.act_func(layer(feat))
        pred_z_so3 = feat[:, :3*256].reshape(batch_size, -1, 3)
        pred_z_inv = feat[:, 3*256:]
        # assert pred_z_so3.shape[1:] == (256, 3), pred_z_so3.shape
        # assert pred_z_inv.shape[1:] == (256,), pred_z_inv.shape
        
        return torch.concat([pred_z_so3, pred_z_inv.unsqueeze(2)], dim=2)

from vec_layers_out import VecLinearNormalizeActivate as VecLNA
from vec_layers_out import VecLinear
class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, max_freq, num_bands, std=0.2):
        super(LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim*2
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2*num_bands
        
        # input_dim = latent_dim[0] * latent_dim[1] + self.timestep_embed.num_bands * 2  # Additional dims for time embedding
        leak_neg_slope=0.2
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(VecLNA(self.latent_dim+self.t_emb_dim, hidden_dims[0], mode="so3", act_func=act_func))
            else:
                self.layers.append(VecLNA(hidden_dims[i-1], hidden_dims[i], mode="so3", act_func=act_func))

        # 最后一层输出与原始潜在维度相同
        # self.layers.append(VecLNA(hidden_dims[-1], self.latent_dim, mode="so3", act_func=act_func))
        self.layers.append(VecLinear(v_in=hidden_dims[-1], v_out=self.latent_dim, mode="so3"))

    def forward(self, x, t):
        batch_size = x.size(0)
        # Embed time
        t_emb = self.timestep_embed(t).unsqueeze(2).repeat(1, 1, 3)
        # t_emb = t.unsqueeze(1).unsqueeze(1).repeat(1, self.t_emb_dim, 3)
        # Concatenate time embedding with the latent code
        x = torch.cat((x, t_emb), dim=1)

        # 显式地遍历层，并在某些层上应用残差连接
        for i, layer in enumerate(self.layers):
            # if i > 0 and isinstance(layer, nn.Linear) and x.size(1) == layer.out_features:
            #     # 应用残差连接
            #     x = x + layer(x)
            # else:
            x = layer(x)
        
        # Reshape the output to the original shape
        return x.view(batch_size, self.latent_dim,3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        print("Debug mode is enabled")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    config = TrainingConfig()
    seed = 0
    torch.manual_seed(seed)


    train_ds = CustomDataset(mode="seperate", normalize_level=1)
    train_dl = DataLoader(train_ds, batch_size=149*2, shuffle=True)
    
    # Create the model and optimizer
    latent_dim = 256
    # hidden_dims = [2048,4096, 8192, 8192,4096, 2048]
    # hidden_dims = [1024,1024,1024,1024,1024,1024]
    # hidden_dims = [256,256,256]
    # hidden_dims = [2048, 4096, 4096, 2048]
    # hidden_dims = [4096, 8192, 8192,4096]
    hidden_dims = [8192,8192*2,8192*2,8192]
    max_freq = 4  # Example max frequency for Fourier features
    num_bands = 4  # Number of frequency bands

    # 初始化模型
    model = origin_LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    args.run_name = "origin_x10_8192_8192*2_normal1_1500k_bs2"
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