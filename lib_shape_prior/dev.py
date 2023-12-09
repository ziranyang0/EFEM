# %%
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
diffusion_ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/vnhead_testresidual_2048*4_cos_200k/model.pth"
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
from core.models import get_model
from dataset import get_dataset
from init import get_cfg, setup_seed, dev_get_cfg

# preparer configuration
category = "mugs"
# category = "kit4cates"
# category = "chairs"
cfg  =dev_get_cfg(category)
# prepare models
ModelClass = get_model(cfg["model"]["model_name"])
model = ModelClass(cfg)
ckpt_path = f"/home/ziran/se3/EFEM/weights/{category}.pt"
ckpt = torch.load(ckpt_path)
model.network.load_state_dict(ckpt['model_state_dict'])
model.network.to(device)

# prepare dataset
DatasetClass = get_dataset(cfg)
datasets_dict = dict()
for mode in cfg["modes"]:
    datasets_dict[mode] = DatasetClass(cfg, mode=mode)

train_ds = datasets_dict["train"]

# %%
bs = 1
model.network.requires_grad_(True)

def get_recon_score(x, s, query, query_mask):
    with torch.enable_grad():
        model.network.zero_grad()
        
        pred_so3_feat, pred_inv_feat = x.detach().requires_grad_(), s.detach().requires_grad_()

        pred_scale = (torch.ones(bs) + 0.2).to(device)
        pred_center = torch.zeros(bs, 1, 3).to(device)
        
        embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            "t": pred_center, # [B, 1, 3]
        }

        sdf_hat = model.network.decode(
            query,
            None,
            embedding,
            return_sdf=True,
        )
        
        sdf_gt = torch.zeros_like(query_mask, device=device, dtype=torch.float32)
        loss = F.mse_loss(sdf_hat[query_mask], sdf_gt[query_mask], reduction='mean')

        loss.backward()
        # grad_so3 = torch.autograd.grad(loss, pred_so3_feat)[0].detach().requires_grad_(False)
        # grad_inv = torch.autograd.grad(loss, pred_inv_feat)[0].detach()

        grad_so3 = pred_so3_feat.grad.detach().requires_grad_(False)
        grad_inv = pred_inv_feat.grad.detach().requires_grad_(False)

        # 清理不再需要的变量和显存
        del pred_so3_feat, pred_inv_feat, sdf_hat, loss, embedding, pred_scale, pred_center

        torch.cuda.empty_cache()

        return grad_so3, grad_inv


# %%
train_loader = DataLoader(
    dataset=train_ds, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True
)
# 获取第一个批次
batch = next(iter(train_loader))

query = batch[0]['inputs'].to(device=device, dtype=torch.float32)
query_mask = (1-batch[0]["inputs_outlier_mask"]).bool()
from tqdm.notebook import tqdm, trange


# %%
x = torch.randn([bs, latent_dim, 3], device=device, requires_grad=True)
s = torch.randn([bs, latent_dim], device=device, requires_grad=True)


# def sample_xs(diffusion_model, x, s, steps, eta):
#     """Draws samples from a diffusion_model given starting noise."""
score_alpha = 1.


ts = x.new_ones([x.shape[0]])

# Create the noise schedule
t = torch.linspace(1, 0, steps + 1)[:-1]
log_snrs = get_ddpm_schedule(t)
alphas, sigmas = get_alphas_sigmas(log_snrs)

# The sampling loop
with torch.no_grad():
    for i in trange(steps):

        # Get the diffusion_model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v_x, v_s = diffusion_model(x, s, ts * log_snrs[i])
            print("v_x: ", v_x.shape, v_x.mean())
            print("v_s: ", v_s.shape, v_s.mean())
        
        with torch.cuda.amp.autocast():
            score_x, score_s = get_recon_score(x, s, query, query_mask)
            print("score_x: ", score_x.shape, score_x.mean())
            print("score_s: ", score_s.shape, score_s.mean())

        v_x = v_x + score_alpha * score_x
        v_s = v_s + score_alpha * score_s
        
        # del score_x, score_s
        # torch.cuda.empty_cache()
        
        # Predict the noise and the denoised image
        pred_x = x * alphas[i] - v_x * sigmas[i]
        eps_x = x * sigmas[i] + v_x * alphas[i]
        pred_s = s * alphas[i] - v_s * sigmas[i]
        eps_s = s * sigmas[i] + v_s * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred_x * alphas[i + 1] + eps_x * adjusted_sigma
            s = pred_s * alphas[i + 1] + eps_s * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma
                s += torch.randn_like(s) * ddim_sigma


