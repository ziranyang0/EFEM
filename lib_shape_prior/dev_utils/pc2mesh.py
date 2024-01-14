path = "/home/ziran/se3/EFEM/data/ShapeNetV1_SDF/03797390/1a1c0a8d4bad82169f0594e65f756cf5/pointcloud.npz"
import numpy as np

# 加载 .npz 文件
data = np.load(path)
# 查看文件中的内容
print("Keys in the NPZ file:", data.files)

# 检查每个键的数据类型和形状
for key in data.files:
    print(f"Key: {key}")
    print("Data type:", type(data[key]))
    print("Shape:", data[key].shape)
    print("---")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
points = data['points']

# 随机选取点云的一个子集
# 例如，选取总点数的 10%
sample_size = int(len(points) * 1)
indices = np.random.choice(len(points), sample_size, replace=False)
points = points[indices]

# # 创建 3D 图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制点云
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# # 设置坐标轴标签
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')

# # 显示图形
# plt.show()

import open3d as o3d
import numpy as np

# 加载点云数据
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Step 2: Estimate normals
point_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 3: Create mesh (using Ball Pivoting here)
# radii = [0.005, 0.01, 0.02, 0.04]
# radii = [0.02, 0.03, 0.04] 
radii = [0.01, 0.02, 0.04] 
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
       point_cloud, o3d.utility.DoubleVector(radii))

# Step 4: Save mesh
o3d.io.write_triangle_mesh("output_mesh.obj", mesh)

