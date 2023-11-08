# %%
# %load_ext autoreload
# %autoreload 2
import torch
import numpy as np

from dataset import get_dataset
from logger import Logger
from core.models import get_model
from core import solver_dict
from init import get_cfg, setup_seed, dev_get_cfg

# preparer configuration
# cfg = get_cfg()
cfg  = dev_get_cfg()


# %%

# set random seed
setup_seed(cfg["rand_seed"])

# prepare dataset
DatasetClass = get_dataset(cfg)
datasets_dict = dict()
for mode in cfg["modes"]:
    datasets_dict[mode] = DatasetClass(cfg, mode=mode)

# prepare models
ModelClass = get_model(cfg["model"]["model_name"])
model = ModelClass(cfg)

# prepare logger
logger = Logger(cfg)

# register dataset, models, logger to the solver
solver = solver_dict[cfg["runner"].lower()](cfg, model, datasets_dict, logger)

# %%

ckpt_path = "/home/ziran/se3/EFEM/lib_shape_prior/log/shape_prior_mugs_old/shape_prior_mugs_dup_old_rename_at_2023-11-05-21-20-55/checkpoint/141_latest.pt"
ckpt = torch.load(ckpt_path)


# 注意不是model.load_state_dict,
# 参见 lib_shape_prior/core/solver_v2.py, lib_shape_prior/core/models/model_base.py

model.network.load_state_dict(ckpt['model_state_dict'])

# %%

model.network.state_dict().keys()

# %%
codebook_path = "/home/ziran/se3/EFEM/lib_shape_prior/mugs.npz"
with np.load(codebook_path) as data:
    # 将 npz 文件内容转换为字典
    codebook = {key: data[key] for key in data}

print(codebook.keys())
for k, v in codebook.items():
    print(k, v.shape)


# %%
bs = 4
pred_so3_feat = codebook['z_so3'][:bs]
pred_inv_feat = codebook['z_inv'][:bs]
pred_scale = codebook['scale'][:bs]
pred_center = codebook['center'][:bs]
query = codebook['pcl'][:bs]

# %%
device = "cuda:0"

N = 16

space_dim = [N, N, N]  # 示例为一个50x50x50的网格

# 创建一个网格，这里我们使用np.linspace来产生线性间隔的点
x = np.linspace(-1, 1, space_dim[0])
y = np.linspace(-1, 1, space_dim[1])
z = np.linspace(-1, 1, space_dim[2])

# 用np.meshgrid得到每个维度的点阵
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 将这些点整理成query的形式，每行是一个点的坐标
query = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

query = torch.tensor(query,dtype=torch.float32).to(device)
query = query.repeat(bs, 1, 1)
query.shape

# %%
# query = codebook['pcl'][:bs]
# query = torch.tensor(query).float().to(device)
# query.shape

# %%
pred_so3_feat = torch.tensor(pred_so3_feat).float().to(device)
pred_inv_feat = torch.tensor(pred_inv_feat).float().to(device)
pred_scale = torch.tensor(pred_scale).float().to(device)
pred_center = torch.tensor(pred_center).float().to(device)

pred_so3_feat.shape


# %%





# query = torch.cat([input_pack["points.uni"], 
#                    input_pack["points.nss"]], dim=1)
embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            # "t": centroid.unsqueeze(1), # [B, 1, 3]
            "t": pred_center, # [B, 1, 3]
        }
sdf_hat = model.network.decode(  # SDF must have nss sampling
            query,
            None,
            embedding,
            return_sdf=True,
        )

# %%
query = codebook['pcl'][:bs]

query = torch.tensor(query).float().to(device)
query.shape

# %%
sdf_hat.shape

# %%
sdf_hat[0]


