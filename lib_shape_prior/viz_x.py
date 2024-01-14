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
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

def viz_pc(input_array):
    if isinstance(input_array, torch.Tensor):
        x, y, z = input_array[:, 0].cpu(), input_array[:, 1].cpu(), input_array[:, 2].cpu()
    elif isinstance(input_array, np.ndarray):
        x, y, z = input_array[:, 0], input_array[:, 1], input_array[:, 2]
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


def viz_x(latent_x, AEmodel, device, normal_params=None, gt_pc=None):
    pred_so3_feat = latent_x[:,:,:3].to(device)
    pred_inv_feat = latent_x[:,:,3].to(device)
    if normal_params is not None:
        print("use normal params:", normal_params)
        pred_so3_feat = pred_so3_feat * normal_params["z_so3_std"] + normal_params["z_so3_mean"]
        pred_inv_feat = pred_inv_feat * normal_params["z_inv_std"] + normal_params["z_inv_mean"]
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

    if gt_pc is not None:
        if isinstance(gt_pc, torch.Tensor):
            gt_pc = gt_pc.cpu().detach().numpy()
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
        print("x max:", verts[:,0].max(), "x min:", verts[:,0].min())
        print("y max:", verts[:,1].max(), "y min:", verts[:,1].min())
        print("z max:", verts[:,2].max(), "z min:", verts[:,2].min())

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

        if gt_pc is not None:
            # 创建点云图表
            point_cloud = go.Scatter3d(
                x=gt_pc[:, 0], y=gt_pc[:, 1], z=gt_pc[:, 2],
                mode='markers',
                marker=dict(
                    size=2,  # 点的大小
                    color='red',  # 点的颜色
                    opacity=0.8
                ),
                name='Point Cloud'
            )
        
        # 合并图表并显示
        if gt_pc is not None:
            fig = go.Figure(data=[mesh, point_cloud], layout=layout)
        else:
            fig = go.Figure(data=[mesh], layout=layout)
        plotly.offline.iplot(fig)


def viz_sdf(sdf, N=50, di = 0.5):
    
    for data in sdf:
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
        print("x max:", verts[:,0].max(), "x min:", verts[:,0].min())
        print("y max:", verts[:,1].max(), "y min:", verts[:,1].min())
        print("z max:", verts[:,2].max(), "z min:", verts[:,2].min())
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

def viz_x_img(latent_x, AEmodel, device, normal_params=None):
    pred_so3_feat = latent_x[:,:,:3].to(device)
    pred_inv_feat = latent_x[:,:,3].to(device)
    if normal_params is not None:
        print("use normal params:", normal_params)
        pred_so3_feat = pred_so3_feat * normal_params["z_so3_std"] + normal_params["z_so3_mean"]
        pred_inv_feat = pred_inv_feat * normal_params["z_inv_std"] + normal_params["z_inv_mean"]
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

    single_images = []  # 用于存储单个图像

    for index, data in enumerate(sdf_grid):
        # 使用 Marching Cubes 算法提取等值面
        # print("Max:", data.max(), "Min", data.min())
        anchor_surface_threshold = 0.00
        if data.min()>anchor_surface_threshold:
            surface_threshold = data.min()+0.001
        elif data.max()<anchor_surface_threshold:
            surface_threshold = data.max()-0.001
        else:
            surface_threshold = anchor_surface_threshold
        verts, faces, normals, values = measure.marching_cubes(data, level=surface_threshold)
        verts = (verts / (N - 1)) * (2*di) - di

        # 创建绘图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制mesh
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
        # 在图像上显示最大值和最小值
        ax.text2D(0.05, 0.95, f"Max: {data.max():.2f}\nMin: {data.min():.2f}", transform=ax.transAxes)

        # 保存图像到缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        single_images.append(im.copy())  # 使用图像的副本
        buf.close()
        plt.close(fig)

    import math

    def calculate_rows_cols(num_images):
        rows = int(math.sqrt(num_images))
        cols = num_images // rows
        # 如果乘积小于总数，则增加一行
        if rows * cols < num_images:
            rows += 1
        return rows, cols

    # 示例：假设有10个图像
    num_images = len(single_images)
    rows, cols = calculate_rows_cols(num_images)
    print("rows:", rows, "cols:", cols)

    # 计算每行的宽度和总体高度
    row_width = sum(im.size[0] for im in single_images[:cols])
    total_height = max(im.size[1] for im in single_images) * rows

    # 创建一个新的图像以拼接所有图像
    combined_image = Image.new('RGB', (row_width, total_height))

    x_offset = 0
    y_offset = 0
    for index, im in enumerate(single_images):
        combined_image.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        if (index + 1) % cols == 0:  # 每行结束时
            x_offset = 0
            y_offset += im.size[1]

    # 返回单个图像的列表和拼接后的大图
    return single_images, combined_image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def viz_code(code_1, code_2= None):
    if isinstance(code_1, torch.Tensor):
        code_1 = code_1.cpu().detach().numpy()
    if isinstance(code_2, torch.Tensor):
        code_2 = code_2.cpu().detach().numpy()
    if code_2 is None:
        len_code = len(code_1) if isinstance(code_1, list) else code_1.shape[0]
        for idx in range(0, len_code):
            plt.figure(figsize=(10, 5))
            sns.heatmap(code_1[idx][0], cmap='viridis', vmin=-0.5, vmax=0.5)
            plt.title(f'idx {idx}: concate(z_so3, z_inv)')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
    else:
        raise NotImplemented


def rotation_matrices(axes, thetas):
    """
    Generate a list of rotation matrices, each associated with counterclockwise 
    rotation about the given axes by the corresponding theta degrees.
    """
    matrices = []
    for axis, theta in zip(axes, thetas):
        theta = np.radians(theta)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                           [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                           [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=np.float32)
        matrices.append(matrix)

    return matrices