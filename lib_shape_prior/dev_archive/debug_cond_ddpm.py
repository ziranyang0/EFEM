# %%
import numpy as np
import torch
from tqdm.notebook import tqdm, trange

from ddpm import LatentDiffusionModel, CustomDataset, get_ddpm_scheduler_variables, extract

timesteps = 1000
timesteps, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = get_ddpm_scheduler_variables(timesteps=timesteps)

# %%

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.manual_seed(1984)

# 初始化模型
latent_dim = 256
hidden_dims = [2048, 2048, 2048, 2048] 
max_freq = 4 
num_bands = 4 
scalar_hidden_dims = [256,256,256,256]
diffusion_model = LatentDiffusionModel(latent_dim, hidden_dims, scalar_hidden_dims, max_freq, num_bands).to(device)

# 12-31
diffusion_ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/mugs_ddpm_cos_200k_l1huber_normT_2048_dataNorm/model_epo199999.pth"
diffusion_ckpt = torch.load(diffusion_ckpt_path)
diffusion_model.load_state_dict(diffusion_ckpt['model'])
diffusion_model = diffusion_model.to(device)
print('Diffusion Model parameters:', sum(p.numel() for p in diffusion_model.parameters()))

# %%
from init import get_AEmodel_cfg
from core.models import get_model
category = "mugs"
config_Only_model = get_AEmodel_cfg()
ModelClass = get_model(config_Only_model["model"]["model_name"])
model = ModelClass(config_Only_model)
ckpt_path = f"/home/ziran/se3/EFEM/weights/{category}.pt"
ckpt = torch.load(ckpt_path)
model.network.load_state_dict(ckpt['model_state_dict'])
model.network = model.network.to(device)

codebook_path = f"/home/ziran/se3/EFEM/cache/mugs.npz"
train_ds = CustomDataset(codebook_path, normalization=True)

bs = 3

# %%
model.network.requires_grad_(True)

def clamp(x, sigma):
    return torch.clamp(x, -sigma, sigma)

SIGMA = 0.1

loss_l1 = torch.nn.L1Loss()
loss_mse = torch.nn.MSELoss()

pred_scale = (torch.ones(bs) + 0.2).to(device)
pred_center = torch.zeros(bs, 1, 3).to(device)

def get_recon_score(x, query, query_mask, sdf_gt = None, normal_params=None):
    with torch.enable_grad():
        model.network.zero_grad()
        x = x.detach().requires_grad_(True)
        pred_so3_feat, pred_inv_feat = x[:,:,:3], x[:,:,3]
        if normal_params is not None:
            pred_so3_feat = pred_so3_feat * normal_params["z_so3_std"] + normal_params["z_so3_mean"]
            pred_inv_feat = pred_inv_feat * normal_params["z_inv_std"] + normal_params["z_inv_mean"]
        embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            "t": pred_center, # [B, 1, 3]
        }
        sdf_hat = model.network.decode(query, None, embedding, return_sdf=True)        
        if sdf_gt is None:
            sdf_gt = torch.zeros_like(query, device=device, dtype=torch.float32)
        # loss = loss_l1(sdf_hat, sdf_gt)
        loss = clamp(sdf_hat - sdf_gt, SIGMA).abs().mean()

        loss_item = loss.item()
        grad_x = torch.autograd.grad(loss, x)[0].detach().requires_grad_(False)
        return grad_x, loss_item
    

x_gt = train_ds[2][0].unsqueeze(0).repeat(bs, 1, 1).to(device=device, dtype=torch.float32)

def get_pc_sdf(mode, bs, uniform_N, surface_N):
    def get_fixed_query_pc_sdf(bs, N = 10, di = 0.5, normal_params=None):
        space_dim = [N, N, N]
        x = np.linspace(-di, di, space_dim[0])
        y = np.linspace(-di, di, space_dim[1])
        z = np.linspace(-di, di, space_dim[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        viz_query = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        viz_query = torch.tensor(viz_query,dtype=torch.float32).to(device)
        viz_query = viz_query.repeat(bs, 1, 1)
        with torch.no_grad():
            pred_so3_feat, pred_inv_feat = x_gt[:,:,:3].detach(), x_gt[:,:,3].detach()
            if normal_params is not None:
                pred_so3_feat = pred_so3_feat * normal_params["z_so3_std"] + normal_params["z_so3_mean"]
                pred_inv_feat = pred_inv_feat * normal_params["z_inv_std"] + normal_params["z_inv_mean"]
            embedding = {
                "z_so3": pred_so3_feat, # [B, 256, 3]
                "z_inv": pred_inv_feat, # [B, 256]
                "s": pred_scale, # [B]
                "t": pred_center, # [B, 1, 3]
            }
            sdf_gt = model.network.decode(viz_query, None, embedding, return_sdf=True)  
        return viz_query, sdf_gt
    
    def sdf2pc(sdf, N, di):
        from skimage import measure
        verts, faces, normals, values = measure.marching_cubes(sdf[0].cpu().numpy().reshape(N,N,N), level=0.)
        verts = (verts / (N - 1)) * (2*di) - di
        return verts
    
    if "fixed" == mode:
        fixed_pc, fixed_sdf = get_fixed_query_pc_sdf(bs, uniform_N, normal_params=train_ds.normal_params)
        return fixed_pc, fixed_sdf
    if "surface" == mode:
        decode_pc, decode_sdf = get_fixed_query_pc_sdf(bs, 60, 0.5, normal_params=train_ds.normal_params)
        verts = sdf2pc(decode_sdf, 60, 0.5)
        surface_pc = torch.tensor(verts).unsqueeze(0).repeat(bs,1,1).to(device)
        surface_sdf = torch.zeros_like(surface_pc[:, :, 0]).to(device)
        return surface_pc, surface_sdf
    if mode == "hybrid":
        if uniform_N >0:
            fixed_pc, fixed_sdf = get_fixed_query_pc_sdf(bs, uniform_N, normal_params=train_ds.normal_params)
            print("fixed_pc:", fixed_pc.shape,"fixed_sdf:", fixed_sdf.shape)

        if surface_N > 0:
            decode_pc, decode_sdf = get_fixed_query_pc_sdf(bs, 60, 0.5, normal_params=train_ds.normal_params)
            verts = sdf2pc(decode_sdf, 60, 0.5)
            surface_pc = torch.tensor(verts).unsqueeze(0).repeat(bs,1,1).to(device)
            B = surface_pc.shape[0]
            indices = torch.randint(60, (B, surface_N))
            surface_pc = torch.stack([surface_pc[b, indices[b]] for b in range(B)])
            surface_sdf = torch.zeros_like(surface_pc[:, :, 0]).to(device)
            print("surface_pc:", surface_pc.shape, "surface_sdf:", surface_sdf.shape)
        
        if uniform_N > 0 and surface_N > 0:
            hybrid_pc = torch.concat([fixed_pc, surface_pc], dim=1)
            hybrid_sdf = torch.concat([fixed_sdf, surface_sdf], dim=1)
            print("hybrid_pc:", hybrid_pc.shape, "hybrid_sdf:", hybrid_sdf.shape)
        elif uniform_N > 0:
            hybrid_pc = fixed_pc
            hybrid_sdf = fixed_sdf
        elif surface_N > 0:
            hybrid_pc = surface_pc
            hybrid_sdf = surface_sdf
        else:
            raise NotImplementedError
        return hybrid_pc, hybrid_sdf

# %%
hybrid_pc, hybrid_sdf = get_pc_sdf("hybrid", bs, 5, 100)



# %%
gamma = 1e4

score_x_list = []
score_recon_list = []
score_list = []
recon_loss_list = []

pred_x_0_list = []
# reverse diffusion
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Note: \grad_{x_t} \log p(x_t|x_0) = - (\epsilon) / (\sqrt{1 - \alpha^{hat}_t})
    #                                   = - model(x, t) / sqrt_one_minus_alphas_cumprod_t
    model_out = model(x, t, timesteps)
    score_x = model_out / sqrt_one_minus_alphas_cumprod_t
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x.shape)
    pred_x_0 = (x - sqrt_one_minus_alphas_cumprod_t * model_out) / sqrt_alphas_cumprod_t
    pred_x_0_list.append(pred_x_0)
    
    score_recon, recon_loss = torch.zeros_like(score_x), 0

    # score_recon, recon_loss = get_recon_score(pred_x_0, query, query_mask, sdf_gt=sdf_values, normal_params=train_ds.normal_params)
    # score_recon, recon_loss = get_recon_score(pred_x_0, verts_tensor, query_mask, sdf_gt=vert_sdf, normal_params=train_ds.normal_params)
    # score_recon, recon_loss = get_recon_score(pred_x_0, fixed_query_pc, query_mask, sdf_gt=sdf_gt, normal_params=train_ds.normal_params)


    
    # score_recon, recon_loss = get_feature_recon_loss(x, x_gt)
    # score_recon, recon_loss = get_feature_recon_AEloss(pred_x_0, x_gt, fixed_pc=True, normal_params=train_ds.normal_params)
    
    score = score_x + gamma * score_recon

    score_x_list.append(score_x.abs().mean().item())
    score_recon_list.append(score_recon.abs().mean().item())
    score_list.append(score.abs().mean().item())
    recon_loss_list.append(recon_loss)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * score
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # noise = gamma * recon_loss
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2:
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    x_t = torch.randn(shape, device=device)
    # traj = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        x_t_minus1 = p_sample(model, x_t, torch.full((b,), i, device=device, dtype=torch.long), i)
        x_t = x_t_minus1
        # traj.append(x_t.cpu().numpy())
    return x_t

torch.manual_seed(10)
x_0 = p_sample_loop(model=diffusion_model, shape=(bs, 256, 4), )

# %%
begin = 0
end = 100
torch.tensor(score_x_list[begin:end]).mean(), torch.tensor(score_recon_list[begin:end]).mean()

# %%
score_x_list

# %%
import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(data, threshold_factor=1.5):
    # q3, q1 = np.percentile(data, [75, 25])
    # iqr = q3 - q1
    # upper_bound = q3 + threshold_factor * iqr
    # return [x for x in data if x <= upper_bound]
    return data


plt.figure(figsize=(12, 8))
plt.plot(remove_outliers(recon_loss_list), label='loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(remove_outliers(score_x_list), label='score_x.abs().mean()')
plt.plot(remove_outliers(score_list), label='score.abs().mean()')
plt.legend()

plt.figure(figsize=(12, 8))

plt.plot(remove_outliers(score_recon_list), label='score_recon.abs().mean()')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Values Over Diffusion Steps')
plt.legend()
# plt.ylim(0, 10)
plt.show()
