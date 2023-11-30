import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    # flash_attn = True
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
).to(device)

training_images = torch.rand(8, 3, 128, 128).to(device) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)