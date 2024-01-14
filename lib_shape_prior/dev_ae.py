# %% [markdown]
# ## Sanity Check: autoencoder fitting input latent code 

# %%

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.manual_seed(1984)

# %%
from init import get_AEmodel_cfg
from core.models import get_model
category = "mugs"
# category = "kit4cates"
# category = "chairs"
config_Only_model = get_AEmodel_cfg()
ModelClass = get_model(config_Only_model["model"]["model_name"])
model = ModelClass(config_Only_model)
ckpt_path = f"/home/ziran/se3/EFEM/weights/{category}.pt"
ckpt = torch.load(ckpt_path)
model.network.load_state_dict(ckpt['model_state_dict'])
model.network = model.network.to(device)

bs = 1


# %%
import pickle
with open(f'./dev_utils/test.sdf', 'rb') as f:
    result = pickle.load(f)
# 分离点云位置和SDF值
points = result['sdf'][:, :3]  # 取前三列为点云位置
sdf_values = result['sdf'][:, 3]  # 取第四列为SDF值
normals = result['normals']
# 检查结果
print("Points shape:", points.shape)
print("SDF Values shape:", sdf_values.shape)
print("Normals shape:", normals.shape)

query = torch.from_numpy(points).unsqueeze(0).repeat(bs, 1, 1).to(device=device, dtype=torch.float32)
sdf_values = torch.from_numpy(sdf_values).unsqueeze(0).repeat(bs, 1).to(device=device, dtype=torch.float32)
query_mask = torch.ones(query.shape[:2]).bool()
normals = torch.from_numpy(normals).unsqueeze(0).repeat(bs, 1, 1).to(device=device, dtype=torch.float32)
print("query max,min: ",query.max(), query.min())
print("query_sdf max,min: ",sdf_values.max(), sdf_values.min())
query.shape, query_mask.shape, sdf_values.shape

# %%
model.network.requires_grad_(True)

def clamp(x, sigma):
    return torch.clamp(x, -sigma, sigma)

SIGMA = 0.1

loss_l1 = torch.nn.L1Loss()
loss_mse = torch.nn.MSELoss()

pred_scale = (torch.ones(bs) + 0.2).to(device)
pred_center = torch.zeros(bs, 1, 3).to(device)

def compute_normals_from_sdf(sdf, query_points):
    # Assumes sdf and query_points are PyTorch tensors
    # sdf: [B, N] - SDF values at query points
    # query_points: [B, N, 3] - Query points (x, y, z)

    # Enable gradient computation for query points
    # query_points.requires_grad_(True)

    # Compute the gradient for the entire batch at once
    grad_outputs = torch.ones_like(sdf, device=device)
    gradients = torch.autograd.grad(outputs=sdf, inputs=query_points,
                                    grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Normalize the gradients to get unit normals
    normals = torch.nn.functional.normalize(gradients, p=2, dim=2)

    return normals

def normal_vector_loss(pred_normals, gt_normals):
    # Calculate the loss between predicted normals and ground truth normals
    # This could be as simple as a mean squared error, or a more complex angular loss
    loss = torch.mean((pred_normals - gt_normals) ** 2)
    return loss


lambda_nv = 0.1
def get_recon_score(x, query, query_mask, sdf_gt = None, usegradloss=False):
    model.network.zero_grad()
    # x = x.detach().requires_grad_(True)
    query.requires_grad_(True)
    pred_so3_feat, pred_inv_feat = x[:,:,:3], x[:,:,3]
    
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
    if sdf_gt is None:
        sdf_gt = torch.zeros_like(query_mask, device=device, dtype=torch.float32)
        
    # Calculate predicted normals from sdf_hat (using automatic differentiation)
    # pred_normals = compute_normals_from_sdf(sdf_hat, query)

    # Calculate the normal vector loss
    # nv_loss = normal_vector_loss(pred_normals, normals)

    # Combine with existing loss
    # total_loss = loss_l1(sdf_hat, sdf_gt) + lambda_nv * nv_loss
    total_loss = loss_l1(sdf_hat, sdf_gt) 
    
    # sdf_hat = clamp(sdf_hat[query_mask], SIGMA)
    # sdf_gt = clamp(sdf_gt[query_mask], SIGMA)

    # loss = F.mse_loss(sdf_hat[query_mask], sdf_gt[query_mask], reduction='mean')
    # loss = F.mse_loss(sdf_hat, sdf_gt, reduction='mean')
    # loss = loss_l1(sdf_hat, sdf_gt)
    return total_loss

def get_fixed_query_pc(bs):
    N = 50
    space_dim = [N, N, N]  # 示例为一个50x50x50的网格
    di = 0.5
    x = np.linspace(-di, di, space_dim[0])
    y = np.linspace(-di, di, space_dim[1])
    z = np.linspace(-di, di, space_dim[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    viz_query = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    viz_query = torch.tensor(viz_query,dtype=torch.float32).to(device)
    viz_query = viz_query.repeat(bs, 1, 1)
    return viz_query
fixed_query_pc = get_fixed_query_pc(bs).to(device)

def get_feature_recon_AEloss(x, x_gt, fixed_pc=True):
    with torch.no_grad():
    # with torch.enable_grad():
        if fixed_pc:
            viz_query = fixed_query_pc.detach().to(device).requires_grad_(True)
        else:
            viz_query = torch.rand(bs, 50*50*50,3).to(device) - 0.5
            viz_query = viz_query.detach().requires_grad_(True)
        pred_so3_feat, pred_inv_feat = x_gt[:,:,:3].detach(), x_gt[:,:,3].detach()
        embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            "t": pred_center, # [B, 1, 3]
        }
        sdf_gt = model.network.decode(viz_query, None, embedding, return_sdf=True)        
        # normals_gt = compute_normals_from_sdf(sdf_gt, viz_query)
        
    with torch.enable_grad():
        pred_so3_feat, pred_inv_feat = x[:,:,:3], x[:,:,3]
        embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            "t": pred_center, # [B, 1, 3]
        }
        sdf_hat = model.network.decode(viz_query, None, embedding, return_sdf=True)        
        normals_hat = compute_normals_from_sdf(sdf_hat, viz_query)

        # loss = F.mse_loss(sdf_hat[query_mask], sdf_gt[query_mask], reduction='mean')
        # sdf_hat = clamp(sdf_hat[query_mask], SIGMA)
        sdf_gt = clamp(sdf_gt, SIGMA)
        loss = loss_l1(sdf_hat, sdf_gt) + 0.1 * (normals_hat.norm(2, dim=2) - 1.0).mean()

        return loss



# %%
from ddpm import LatentDiffusionModel, CustomDataset, get_ddpm_scheduler_variables, extract
codebook_path = f"/home/ziran/se3/EFEM/cache/mugs.npz"
train_ds = CustomDataset(codebook_path)

codebook_x = torch.stack([train_ds[idx][0] for idx in range(len(train_ds))], dim=0)

codebook_x_mean = codebook_x.mean(dim=(0, 1)).to(device)
codebook_x_std = codebook_x.std(dim=(0, 1)).to(device)
print(codebook_x_mean.shape, codebook_x_std.shape)
print(codebook_x_mean, codebook_x_std)
# %%

@torch.no_grad()
def eval_code(x, x_gt):
    x = x.detach()
    x_gt = x_gt.detach()
    return torch.mean(torch.abs(x - x_gt))
# %%
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
exp_name = "sanity_AE_step100k_cos_lr5e-4_DecoderSepelr-4_l1_gtclamp_pointcloud_NOnormto1"
wandb.init(project="vnDiffusion", entity="_zrrr", name=exp_name)

# y = torch.rand(bs, 256, 4, device=device) - 0.5
# y = y.clone().detach().requires_grad_(True)  # 这使 y 成为一个叶子节点
# x = torch.rand(bs, 256, 4, device=device) - 0.5
# x = x.clone().detach().requires_grad_(True)  # 这使 y 成为一个叶子节点
x = torch.zeros(bs, 256, 4, requires_grad=True, device=device)
opt_steps = 100000

decoder_params = model.network.network_dict["decoder"].parameters()
optimizer = torch.optim.Adam([x], lr=5e-4)
decoder_optimizer = torch.optim.Adam(list(decoder_params), lr=5e-4)
# optimizer = torch.optim.Adam([x], lr=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=opt_steps, eta_min=0)

# 初始化一个用于存储损失值的列表
log_list = []
x_values = []
x_gt = train_ds[1][0].unsqueeze(0).repeat(bs, 1, 1).to(device)

for i in range(opt_steps):
    # x = y * codebook_x_std + codebook_x_mean
    optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = get_recon_score(x, query, query_mask, sdf_values)
    # loss = get_feature_recon_AEloss(x, x_gt, fixed_pc=True)
    loss.backward()

    optimizer.step()
    decoder_optimizer.step()
    scheduler.step() 
    # 记录损失值
    code_dist_item = eval_code(x, x_gt).item()
    
    loss_item = loss.item()
    grad_x_item = x.grad.detach().abs().mean().item()

    log = {
        "step": i,
        "code_dist": code_dist_item,
        "loss": loss_item,
        "grad_x": grad_x_item,
        }
    log_list.append(log)
    if i % 1 == 0:
        print(log)
        wandb.log(log)
    if (i+1) % 10000 == 0:
        x_values.append(x.detach().cpu().numpy())


import pickle
import os
if not os.path.exists(f"{exp_name}"):
    os.makedirs(f"{exp_name}")

# 保存到文件
with open(f'{exp_name}/x_values.pkl', 'wb') as f:
    pickle.dump(x_values, f)

torch.save(model.network.state_dict(), f'{exp_name}/model_state_dict.pth')
