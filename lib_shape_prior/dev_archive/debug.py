import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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

SIGMA = 1

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
def get_recon_score(x, query, query_mask, sdf_gt = None):
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
    pred_normals = compute_normals_from_sdf(sdf_hat, query)

    # Calculate the normal vector loss
    nv_loss = normal_vector_loss(pred_normals, normals)

    # Combine with existing loss
    total_loss = loss_l1(sdf_hat, sdf_gt) + lambda_nv * nv_loss
    
    # sdf_hat = clamp(sdf_hat[query_mask], SIGMA)
    # sdf_gt = clamp(sdf_gt[query_mask], SIGMA)

    # loss = F.mse_loss(sdf_hat[query_mask], sdf_gt[query_mask], reduction='mean')
    # loss = F.mse_loss(sdf_hat, sdf_gt, reduction='mean')
    # loss = loss_l1(sdf_hat, sdf_gt)
    return total_loss
# %%
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


# y = torch.rand(bs, 256, 4, device=device) - 0.5
# y = y.clone().detach().requires_grad_(True)  # 这使 y 成为一个叶子节点
x = torch.zeros(bs, 256, 4, requires_grad=True, device=device)
opt_steps = 20000

# decoder_params = model.network.network_dict["decoder"].parameters()
# optimizer = torch.optim.Adam([x]+ list(decoder_params), lr=1e-3)
optimizer = torch.optim.Adam([x], lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=opt_steps, eta_min=0)

# 初始化一个用于存储损失值的列表
loss_values = []
grad_x_values = []
x_values = []

for i in range(opt_steps):
    # x = y * codebook_x_std + codebook_x_mean
    optimizer.zero_grad()
    loss = get_recon_score(x, query, query_mask, sdf_values)
    # loss = get_feature_recon_AEloss(x, train_ds[0][0].unsqueeze(0).to(device), fixed_pc=False)
    loss.backward()
    optimizer.step()
    scheduler.step() 
    # 记录损失值
    loss_item = loss.item()
    grad_x_item = x.grad.detach().abs().mean().item()
    loss_values.append(loss_item)
    grad_x_values.append(grad_x_item)

    if i % 100 == 0:
        print(f"i {i}:", loss_item)
        print(f"grad_x abs mean {i}:", grad_x_item)
    if (i+1) % 1000 == 0:
        x_values.append(x.detach().cpu().numpy())


import pickle

# 保存到文件
with open('x_values.pkl', 'wb') as f:
    pickle.dump(x_values, f)

# %%
len(x_values)
x_values[0]

# %%
import pickle
file_path = "/home/ziran/se3/EFEM/lib_shape_prior/sanity_AE_step200k_lr5e-4_cos_points30k/x_values12-20.pkl"
with open(file_path, "r") as f:
    x_values = pickle.load(f)

# %%
len(x_values)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

for idx in range(len(x_values)):
    plt.figure(figsize=(20, 10))
    sns.heatmap(x_values[idx][0], cmap='viridis', vmin=-0.5, vmax=0.5)
    plt.title(f'idx{1} generated code(256x4), concate(z_so3, z_inv)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

# %%


# %%


# %%


# %%

# 绘制损失曲线
plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()


# %%
import torch
from viz_x import viz_x, viz_pc
viz_x(torch.concat([torch.tensor(x_values[idx])for idx in range(len(x_values))], dim=0)[-5:, :,:] , model, device)


