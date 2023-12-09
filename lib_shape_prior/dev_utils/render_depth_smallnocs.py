# 2022.8.6: use larger camera shift for nocs like single view application

# use pyrender to render depth pcl for occnet data

# 2022.7.2 sample the SDF (uni + nss) from DISN water tight mesh, and the aligned with OCC data

# saved_mesh = (ori_mesh.vertices - centroid) / float(m)
# OCC scale: OCC = (ShapenetMesh - loc) / scale

import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path as osp
import pyrender

import time
import numpy as np
import trimesh
import imageio
from transforms3d.euler import euler2mat

import os.path as osp
import glob
import numpy as np
from datetime import datetime
import trimesh
from tqdm import tqdm
from multiprocessing.dummy import Pool

import argparse

THREADS = 20
ERROR_FN = "/home/ziran/se3/EFEM/lib_shape_prior/dev_utils/RENDER_ERROR.txt"

EPS = 1e-10
N_view = 12
SCRIPT_PATH = osp.join(osp.dirname(__file__), "render_depth_thread_small.py")


def wrapper(param):
    start_t = time.time()

    cmd = f"python {SCRIPT_PATH} --mesh '{param[0]}' --dst '{param[1]}' --loc_x {param[2][0]}  --loc_y {param[2][1]}  --loc_z {param[2][2]} --scale {param[3]} --n_view {N_view} --tmp {param[4]}"
    os.system(cmd)
    mesh_fn, dst, loc, scale, tmp = param
    print(cmd)
    if not osp.exists(dst) or len(os.listdir(dst)) < N_view:
        cmd = f"echo 'Error: {mesh_fn} {dst}' >> {ERROR_FN}"
        os.system(cmd)
        print(cmd)
    print(f"{dst} finished in {time.time() - start_t}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_id', default="mugs", help="class")
    parser.add_argument('--src', default="/orion/group/chairs_and_mugs/mugs", help="source")
    parser.add_argument('--dst', default="/orion/group/chairs_and_mugs/mugs/mugs_pair", help="target")
    parser.add_argument('--tmp', default="/orion/group/chairs_and_mugs/mugs/tmp", help="tmp")
    opt = parser.parse_args()

    # MAIN
    PARAM = {}
    
    cates = [f"{opt.class_id}_novel_pile", f"{opt.class_id}_uniform_so3", f"{opt.class_id}_uniform_z"]
    opt.src = f"/orion/group/chairs_and_mugs/{opt.class_id}"
    opt.dst = f"/orion/group/chairs_and_mugs/{opt.class_id}/{opt.class_id}_pair"
    opt.tmp = f"/orion/group/chairs_and_mugs/{opt.class_id}/tmp"
    
    SRC = opt.src
    DST = opt.dst

    print("Preparing params")
    for cate in cates:
        PARAM[cate] = []
        cate_mesh_dir = os.path.join(SRC, cate, "train")
        obj_id_list = os.listdir(cate_mesh_dir)
        for obj_id in tqdm(obj_id_list):
            try:
                if opt.class_id == "chairs":
                    mesh_fn = osp.join(cate_mesh_dir, obj_id, "raster/mesh.obj")
                else:
                    mesh_fn = osp.join(cate_mesh_dir, obj_id, "kuafu/mesh.obj")
                dst = osp.join(
                    DST, cate + "_dep_small", obj_id,
                )  # v2 is only top down; # v3 is random rotation
                loc = np.array([0.0, 0.0, 1.0])
                s = float(1.0)
                param = (mesh_fn, dst, loc, s, osp.join(opt.tmp, cate))
                PARAM[cate].append(param)
            except:
                print(f"{cate} {obj_id} param prepare fails, skip")
        print(f"Cate: {cate} has {len(PARAM[cate])} instances")
    for cate, param_list in PARAM.items():
        print(cate)
        with Pool(THREADS) as p:
            p.map(wrapper, param_list)
    print()