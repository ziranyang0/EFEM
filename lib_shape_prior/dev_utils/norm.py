# %%
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


# %%
import open3d as o3d
import numpy as np

points = data['points']

# 随机选取点云的一个子集
# 例如，选取总点数的 10%
sample_size = int(len(points) * 0.1)
indices = np.random.choice(len(points), sample_size, replace=False)
points = points[indices]


# 将 NumPy 数组转换为 Open3D 的点云格式
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

# 计算法线
point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 获取法线（作为 NumPy 数组）
normals_np = np.asarray(point_cloud_o3d.normals)

# 显示前几个法线
print(normals_np.shape)


# %%
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# 绘制法线
for i in range(len(points)):
    ax.quiver(
        points[i, 0], points[i, 1], points[i, 2],
        normals_np[i, 0], normals_np[i, 1], normals_np[i, 2],
        length=0.1, color='red'
    )

# 设置图形属性
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig('norm.png')
plt.show()

# %%



