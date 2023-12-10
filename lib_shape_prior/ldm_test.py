# %%
from tqdm.notebook import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from dev_diffusionVN_xs import *
from dev_utils import sample_xs



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.manual_seed(1984)



# 初始化模型
latent_dim = 256
hidden_dims = [2048, 2048, 2048, 2048] 
max_freq = 4  # Example max frequency for Fourier features
num_bands = 4  # Number of frequency bands
scalar_hidden_dims = [256,256,256,256]
diffusion_model = LatentDiffusionModel(latent_dim, hidden_dims, scalar_hidden_dims, max_freq, num_bands).to(device)
diffusion_ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/12-9/residual_cos_200k/model_59999.pt"
# diffusion_ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/chairs_vnhead_residual_2048*4_cos_10k/model.pth"
diffusion_ckpt = torch.load(diffusion_ckpt_path)
diffusion_model.load_state_dict(diffusion_ckpt['model'])
diffusion_model = diffusion_model.to(device)
print('Diffusion Model parameters:', sum(p.numel() for p in diffusion_model.parameters()))


scaler = torch.cuda.amp.GradScaler()
rng = torch.quasirandom.SobolEngine(1, scramble=True)
ema_decay = 0.998
steps = 1000
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.

# %%
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# %%
import torch
latent_dim = 256
bs = 4
noise = torch.randn((bs, latent_dim, 4), device=device)
input_tensor = noise.to(device)

for t in noise_scheduler.timesteps:
    with torch.no_grad():
        input_x, input_s = input_tensor.split([3, 1], dim=2)
        t = t.reshape(1).repeat(bs).to(device)
        noisy_residual = diffusion_model(input_x, input_s, t)
    previous_noisy_sample = noise_scheduler.step(noisy_residual, t, input_tensor).prev_sample
    input_tensor = previous_noisy_sample


