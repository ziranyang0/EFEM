import pybullet as p
import numpy as np
from multi_camera import MultiCamera


p.connect(p.DIRECT)  # 或 p.GUI，视您的需求而定

# plane_id = p.loadURDF("plane.urdf")  # 直接加载标准平面模型

# 加载您刚刚创建的 URDF 文件
mesh_id = p.loadURDF("mesh.urdf")



# 初始化相机视图
cam_yaws = [-30, 10, 50, 90, 130, 170, 210]
cam_pitches = [-70, -10, -65, -40, -10, -25, -60]
cam_dist = 0.85
cam_target = np.array([0.35, 0, 0])

your_object_ids = 0
# 渲染图像
rendered_images = MultiCamera.render(
    sim=p,  # PyBullet 实例
    object_ids=[your_object_ids],  # 你加载的物体的ID列表
    cam_yaws=cam_yaws,
    cam_pitches=cam_pitches,
    cam_dist=cam_dist,
    cam_target=cam_target,
    views=[2],  # 选择要渲染的视图索引
    width=100,  # 图像宽度
    height=100  # 图像高度
)

# 获取渲染的图像
images = rendered_images['images']
# 可以进行进一步的处理或显示这些图像
