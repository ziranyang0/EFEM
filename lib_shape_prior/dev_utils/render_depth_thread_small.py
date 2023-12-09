import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path as osp
import pyrender

import time
import numpy as np
import trimesh
import imageio
from transforms3d.euler import euler2mat
from scipy.spatial.transform import Rotation

import os.path as osp
import numpy as np
from datetime import datetime
import trimesh

EPS = 1e-10
RENDER_WH = (256, 256)
camera_translation_bound = np.array([0.6, 0.6, 1.0])[None, :]
camera_center_base = np.array([0.0, 0.0, 1.0])[None, :]


def sample_thread(mesh_fn, dst, loc, scale, N_view, tmp):
    if os.path.exists(dst):
        print(f"{dst} exists, skipped")
        return
    os.makedirs(osp.dirname(dst), exist_ok=True)

    date_time = datetime.now().strftime("%H_%M_%S")
    #mesh_name = osp.basename(mesh_fn)
    mesh_name = mesh_fn.split('/')[-3]
    tmp_dst = osp.join(
        tmp, mesh_name + "_" + date_time + "_processing"
    )
    os.makedirs(tmp_dst, exist_ok=True)

    # normalize the mesh
    mesh = trimesh.load(mesh_fn, process=False)
    mesh.apply_translation(-loc)
    mesh.apply_scale(1.0 / scale)

    # generate poses
    T_list = sample_poses(N_view)
    # pcl_all = []
    for vid, T_cw in enumerate(T_list):
        # p_w = T_cw @ p_c
        rgb, dep, cam_pcl = render(mesh, T_cw)
        world_pcl = (
            T_cw[:3, :3][None, ...] @ cam_pcl[..., None] + T_cw[:3, 3:4][None, ...]
        ).squeeze(-1)
        
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, world_pcl)
        n = mesh.face_normals[triangle_id]
        
        imageio.imsave(osp.join(tmp_dst, f"rgb_{vid}.png"), rgb)
        np.savez_compressed(
            osp.join(tmp_dst, f"dep_pcl_{vid}.npz"),
            p_w=world_pcl,
            T_cw=T_cw,
            n=n,
        )
        # pcl_all.append(world_pcl)
    # np.savetxt(osp.join(tmp_dst, f"all_pts.txt"), np.concatenate(pcl_all, 0))

    os.system(f"cp -r {tmp_dst} {dst+'_processing'}")
    os.system(f"mv {dst+'_processing'} {dst}")
    os.system(f"rm -r {tmp_dst}")

    return


def sample_poses(N):
    # first random shift the camera
    shift = (
        np.random.uniform(0.0, 0.0, size=(N, 3)) * camera_translation_bound + camera_center_base
    )
    # camera pose
    #theta_x = np.random.uniform(np.pi / 4, np.pi / 4, (N))
    #theta_z = [np.pi / 4, np.pi / 4 * 3, np.pi / 4 * 5, np.pi / 4 * 7]
    theta_x = np.random.uniform(np.pi / 6, np.pi / 3, (N))
    theta_z = np.random.uniform(0.0, np.pi * 2, (N))
    R_list = [euler2mat(x, z, 0.0, "sxzy") for x, z in zip(theta_x, theta_z)]
    # ! random rotation
    #R_list = [Rotation.random().as_matrix() for _ in range(N)]
    
    t_list = [(R @ t[:, None]).squeeze(-1) for t, R in zip(shift, R_list)]
    T_list = []
    for R, t in zip(R_list, t_list):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t
        T_list.append(T)

    return T_list


def render(mesh, camera_pose):
    r = pyrender.OffscreenRenderer(RENDER_WH[0], RENDER_WH[1])
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0,
        aspectRatio=1.0,
    )

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    del scene
    fg_mask = depth > 0.0
    z = depth[fg_mask]

    K = camera.get_projection_matrix(width=RENDER_WH[0], height=RENDER_WH[1])
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    gridyy, gridxx = np.mgrid[:h, :w]
    gridyy = (gridyy / h - 0.5) * 2
    gridxx = (gridxx / w - 0.5) * 2

    # here we first negative z, previously we need to negative y because we generate row numbers from top down, the y+ of camera is down-top, so double negative is positive, but x need to neg now
    Z = -depth
    X = -(gridxx) / fx * Z
    Y = (gridyy) / fy * Z
    xyz = np.concatenate((X[..., np.newaxis], Y[..., np.newaxis], Z[..., np.newaxis]), axis=2)
    camera_view_pc = xyz[depth > 0, :]

    return color, depth, camera_view_pc


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="process sequence")
    arg_parser.add_argument("--mesh", type=str)
    arg_parser.add_argument("--dst", type=str)
    arg_parser.add_argument("--loc_x", type=float)
    arg_parser.add_argument("--loc_y", type=float)
    arg_parser.add_argument("--loc_z", type=float)
    arg_parser.add_argument("--scale", type=float)
    arg_parser.add_argument("--n_view", type=int)
    arg_parser.add_argument("--tmp", type=str)
    args = arg_parser.parse_args()
    loc = np.array([args.loc_x, args.loc_y, args.loc_z])
    sample_thread(args.mesh, args.dst, loc, args.scale, args.n_view, args.tmp)