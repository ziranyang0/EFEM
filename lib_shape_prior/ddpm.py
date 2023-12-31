
import numpy as np
import wandb
from inspect import isfunction
from functools import partial

from tqdm.auto import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F





def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_ddpm_scheduler_variables(timesteps):
    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)
    # betas = cosine_beta_schedule(timesteps=timesteps)
    # betas = quadratic_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return timesteps, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

# def get_ddim_scheduler_variables(timesteps):
#     # define beta schedule
#     betas = linear_beta_schedule(timesteps=timesteps)
#     # betas = cosine_beta_schedule(timesteps=timesteps)
#     # betas = quadratic_beta_schedule(timesteps=timesteps)

#     # define alphas
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, axis=0)
#     alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#     sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

#     # calculations for diffusion q(x_t | x_{t-1}) and others
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

#     # calculations for posterior q(x_{t-1} | x_t, x_0)
#     posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#     return timesteps, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

timesteps = 1000
timesteps, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = get_ddpm_scheduler_variables(timesteps=timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# reverse diffusion loss
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, timesteps)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss



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
    
    def forward(self, z, t, timesteps):
        t=t/timesteps
        z_so3, z_inv = z[:,:,:3], z[:,:,3]
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
        
        pred_z_so3, pred_z_inv = self.vn_head(z_so3, z_inv)
        pred = torch.concat([pred_z_so3, pred_z_inv.unsqueeze(-1)], dim=-1)
        
        return pred

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, codebook_path, normalization = False):
        with np.load(codebook_path) as data:
            # 将 npz 文件内容转换为字典
            codebook = {key: data[key] for key in data}
        del codebook['id']
        for k, v in codebook.items():
            if isinstance(v, np.ndarray):
                newv = torch.from_numpy(v)
                codebook[k] = newv
            print(k, v.shape)
        
        self.normalization = normalization
        if normalization:
            z_so3_mean = codebook['z_so3'].mean()
            z_so3_std = codebook['z_so3'].std()
            z_so3_normalize = (codebook['z_so3']-z_so3_mean)/z_so3_std
            z_inv_mean = codebook['z_inv'].mean()
            z_inv_std = codebook['z_inv'].std()
            z_inv_normalize = (codebook['z_inv']-z_inv_mean)/z_inv_std
            codebook['z_so3'] = z_so3_normalize
            codebook['z_inv'] = z_inv_normalize
            print("========== Data Normalization BEGIN ==========")
            print("z_so3_mean", z_so3_mean)
            print("z_so3_std", z_so3_std)
            print("z_inv_mean", z_inv_mean)
            print("z_inv_std", z_inv_std)
            self.normal_params = {
                "z_so3_mean": z_so3_mean,
                "z_so3_std": z_so3_std,
                "z_inv_mean": z_inv_mean,
                "z_inv_std": z_inv_std,
            }
            print("========== Data Normalization FINISHED ==========")
            

        self.x = [codebook['z_so3'][i] for i in range(codebook['z_so3'].shape[0])] # [256, 3]
        self.s = [codebook['z_inv'][i] for i in range(codebook['z_inv'].shape[0])] # [256]
        self.pcl = [codebook['pcl'][i] for i in range(codebook['pcl'].shape[0])]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        s = self.s[idx]
        pcl_item = self.pcl[idx]
        return torch.concat([x, s.unsqueeze(-1)], dim=-1), pcl_item

import os
def save_model(model, epoch,
         args, exp_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt"):
    if not os.path.exists(f"{exp_path}/{args.exp_name}"):
        os.makedirs(f"{exp_path}/{args.exp_name}")
    filename = f'{exp_path}/{args.exp_name}/model_epo{epoch}.pth'
    obj = {
        'model': model.state_dict(),
        'epoch': epoch,
        "category": args.category
    }
    torch.save(obj, filename)

def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(1984)


    import argparse
    args = argparse.Namespace()
    args.category = "mugs"
    args.bs = 149
    args.exp_name = "mugs_ddpm_cos_400k_l1huber_normT_2048_dataNorm"
    args.save_interval = 100000
    args.debug = False
    args.log_interval = 1
    args.num_epochs = 400000



    codebook_path = f"/home/ziran/se3/EFEM/cache/{args.category}.npz"
    # codebook_path = f"/home/ziran/se3/EFEM/lib_shape_prior/log/12_10_shape_prior_mugs_old/12_10_shape_prior_mugs_FOR_hopefullybetterAE/codebook.npz"
    train_ds = CustomDataset(codebook_path, normalization = True)

    # 创建 DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

    # 初始化模型
    latent_dim = 256
    hidden_dims = [2048, 2048, 2048, 2048] 
    # hidden_dims = [1024, 1024, 1024, 1024] 
    max_freq = 4  # Example max frequency for Fourier features
    num_bands = 4  # Number of frequency bands
    scalar_hidden_dims = [256,256,256,256]
    model = LatentDiffusionModel(latent_dim, hidden_dims, scalar_hidden_dims, max_freq, num_bands).to(device)

    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl)*args.num_epochs, eta_min=1e-7)
    if not args.debug:
        wandb.init(project="vnDiffusion", entity="_zrrr", name=args.exp_name)

    print('Diffusion Model parameters:', sum(p.numel() for p in model.parameters()))
    if not args.debug:
        wandb.log({"max_freq": max_freq, "num_bands": num_bands,
                #    "device": device,
                "model_params": sum(p.numel() for p in model.parameters())})


    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dl):
            optimizer.zero_grad()

            batch_size = batch[0].shape[0]
            batch = batch[0].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            # if step % 100 == 0:
            #     print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
            if step % args.log_interval == 0 and epoch % 1 == 0:
                log_msg = {"epoch": epoch, "iteration": step,
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],}
                wandb.log(log_msg)
                print(log_msg)

        if (epoch + 1) % args.save_interval == 0:
            save_model(model, epoch, args)
        scheduler.step()

if __name__ == "__main__":
    main()