import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from ddpm import *
from skimage import measure
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.offline as py

def viz_pc(tensor):
    x, y, z = tensor[:, 0].cpu(), tensor[:, 1].cpu(), tensor[:, 2].cpu()
    # 创建散点图对象
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8
        )
    )
    # 设置布局
    layout = go.Layout(
        title='3D Point Cloud',
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )

    # 组合图表和布局
    fig = go.Figure(data=[trace], layout=layout)

    # 显示图表
    py.iplot(fig)


def viz_x(latent_x, AEmodel, device):
    pred_so3_feat = latent_x[:,:,:3].to(device)
    pred_inv_feat = latent_x[:,:,3].to(device)
    pred_scale = torch.ones((latent_x.shape[0],)).to(device) + 0.2
    pred_center = torch.zeros((latent_x.shape[0], 1, 3)).to(device)+0.
    print("shape: ", pred_so3_feat.shape, pred_inv_feat.shape, pred_scale.shape, pred_center.shape)

    torch.cuda.empty_cache()
    N = 50
    space_dim = [N, N, N]  # 示例为一个50x50x50的网格
    di = 0.5
    # 创建一个网格，这里我们使用np.linspace来产生线性间隔的点
    x = np.linspace(-di, di, space_dim[0])
    y = np.linspace(-di, di, space_dim[1])
    z = np.linspace(-di, di, space_dim[2])

    # 用np.meshgrid得到每个维度的点阵
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 将这些点整理成query的形式，每行是一个点的坐标
    viz_query = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    viz_query = torch.tensor(viz_query,dtype=torch.float32).to(device)
    viz_query = viz_query.repeat(pred_so3_feat.shape[0], 1, 1)
    viz_query.shape

    with torch.no_grad():
        embedding = {
            "z_so3": pred_so3_feat, # [B, 256, 3]
            "z_inv": pred_inv_feat, # [B, 256]
            "s": pred_scale, # [B]
            # "t": centroid.unsqueeze(1), # [B, 1, 3]
            "t": pred_center, # [B, 1, 3]
        }

        sdf_hat = AEmodel.network.decode(  # SDF must have nss sampling
            viz_query,
            None,
            embedding,
            return_sdf=True,
        )
        sdf_grid = sdf_hat.reshape(-1, space_dim[0], space_dim[1], space_dim[2]).to("cpu").detach().numpy()

    for data in sdf_grid:
        plotly.offline.init_notebook_mode()

        # 使用 Marching Cubes 算法提取等值面
        print("Max:", data.max(), "Min", data.min())
        anchor_surface_threshold = 0.00
        if data.min()>anchor_surface_threshold:
            surface_threshold = data.min()+0.001
        elif data.max()<anchor_surface_threshold:
            surface_threshold = data.max()-0.001
        else:
            surface_threshold = anchor_surface_threshold
        verts, faces, normals, values = measure.marching_cubes(data, level=surface_threshold)
        verts = (verts / (N - 1)) * (2*di) - di

        x, y, z = zip(*verts)
        i, j, k = zip(*faces)

        # 创建 mesh3d 图表
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.5,
            name='Mesh'
        )
        # 创建图表布局
        layout = go.Layout(
            title='3D Mesh and Point Cloud Visualization',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            )
        )

        # 合并图表并显示
        # fig = go.Figure(data=[mesh, pointcloud_plot], layout=layout)
        fig = go.Figure(data=[mesh], layout=layout)
        plotly.offline.iplot(fig)



