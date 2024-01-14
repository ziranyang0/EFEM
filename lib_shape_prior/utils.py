import torch
import numpy as np

def get_fixed_query_pc_sdf(model, x_gt, pred_scale, pred_center, N = 10, di = 0.5, normal_params=None):
    space_dim = [N, N, N]
    x = np.linspace(-di, di, space_dim[0])
    y = np.linspace(-di, di, space_dim[1])
    z = np.linspace(-di, di, space_dim[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    viz_query = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    viz_query = torch.tensor(viz_query,dtype=torch.float32).to(x_gt.device)
    viz_query = viz_query.repeat(x_gt.shape[0], 1, 1)
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
    verts_list = []
    normals_list = []
    for sdf_item in sdf:
        verts, faces, normals, values = measure.marching_cubes(sdf[0].cpu().numpy().reshape(N,N,N), level=0.)
        verts = (verts / (N - 1)) * (2*di) - di
        verts_list.append(verts)
        normals_list.append(normals)
    return torch.tensor(np.stack(verts_list, axis=0)), torch.tensor(np.stack(normals_list, axis=0))
    # return torch.stack(verts_list, dim=0), torch.stack(normals_list, dim=0)


import torch
import torch.nn.functional as F


def closest_point_matching(pc1, pc2):
    """
    批次化寻找最近的点对
    """
    B, N1, _ = pc1.shape
    N2 = pc2.shape[1]
    pc1_expand = pc1.unsqueeze(2).expand(B, N1, N2, 3)
    pc2_expand = pc2.unsqueeze(1).expand(B, N1, N2, 3)
    dists = torch.norm(pc1_expand - pc2_expand, dim=3)  # 计算距离
    indices = torch.argmin(dists, dim=2)
    return torch.gather(pc2, 1, indices.unsqueeze(-1).expand(B, N1, 3))

def compute_rotation_matrix(pc1, pc2):
    """
    批次化计算旋转矩阵
    """
    H = torch.matmul(pc1.transpose(1, 2), pc2)  # 计算协方差矩阵
    U, S, V = torch.linalg.svd(H)  # 奇异值分解
    rot = torch.matmul(V, U.transpose(1, 2))  # 计算旋转矩阵

    # 确保旋转矩阵是正交的
    det = torch.det(rot)
    sign_det = torch.sign(det).unsqueeze(-1)  # 形状变为 [B, 1]
    V[:, :, -1] *= sign_det
    rot = torch.matmul(V, U.transpose(1, 2))
    return rot


def icp(pred_pc, gt_pc, iterations=10):
    """
    迭代最近点算法
    """
    for i in range(iterations):
        # matched_gt_pc = closest_point_matching(pred_pc, gt_pc)
        matched_gt_pc = gt_pc
        # print("matched_gt_pc", matched_gt_pc.shape)
        rot = compute_rotation_matrix(pred_pc, matched_gt_pc)
        pred_pc = torch.matmul(pred_pc, rot.transpose(1, 2))  # 应用旋转

    return rot

