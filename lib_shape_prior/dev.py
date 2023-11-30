device = "cuda"

import torch
import numpy as np
from ldm import LatentDiffusionModel
latent_dim = 256
# hidden_dims = [2048,4096, 8192, 8192,4096, 2048]  # Example of hidden dimensions
# hidden_dims = [4096, 8192, 8192,4096]  # Example of hidden dimensions
# hidden_dims = [1024,1024,1024,1024,1024,1024]  # Example of hidden dimensions
# hidden_dims = [256,256,256]  # Example of hidden dimensions
hidden_dims = [2048, 4096, 4096, 2048]
max_freq = 4  # Example max frequency for Fourier features
num_bands = 4  # Number of frequency bands
model = LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/ddpm-butterflies-128/model.pt"
model.load_state_dict(torch.load(ckpt_path)['model'])


from diffusers import DDPMScheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

bs = 1
noise = torch.randn((bs, 512, 3), device=device)

input_tensor = noise

for t in scheduler.timesteps:
    t = t.reshape(1).repeat(bs).to(device)
    with torch.no_grad():
        noisy_residual = model(input_tensor, t)
    noisy_residual = noisy_residual.cpu()
    t = t.cpu()
    input_tensor = input_tensor.cpu()
    previous_noisy_sample = scheduler.step(noisy_residual, t, input_tensor).prev_sample
    input_tensor = previous_noisy_sample.to(device)



fakes = input_tensor
query_start = 0
query_end = 1
pred_so3_feat = fakes[query_start:query_end,:256,].to(device)
pred_inv_feat = fakes[query_start:query_end,256:512,:].mean(dim=2).to(device)
pred_scale = torch.ones((query_end-query_start)) + 0.2
pred_center = torch.zeros(query_end-query_start, 1, 3)

print(pred_so3_feat.shape, pred_inv_feat.shape, pred_scale.shape, pred_center.shape)
pred_so3_feat = torch.tensor(pred_so3_feat).float().to(device)
pred_inv_feat = torch.tensor(pred_inv_feat).float().to(device)
pred_scale = torch.tensor(pred_scale).float().to(device)
pred_center = torch.tensor(pred_center).float().to(device)


codebook_path = "/home/ziran/se3/EFEM/cache/mugs.npz"

with np.load(codebook_path) as data:
    # 将 npz 文件内容转换为字典
    codebook = {key: data[key] for key in data}

del codebook['id']
for k, v in codebook.items():
    if isinstance(v, np.ndarray):
        newv = torch.from_numpy(v)
        codebook[k] = newv
    print(k, v.shape)