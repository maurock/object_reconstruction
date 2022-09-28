import open3d as o3d
import numpy as np
import trimesh
from pterotactyl.utility import utils
from pytorch3d.loss import chamfer_distance as cuda_cd
import torch
"""
Quick script for the report. It draws two meshes with similar shapes but different vertices. It computes the Chamfer Distance.
"""
vertices1 = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0]]))
triangles1 = o3d.utility.Vector3iVector(
        np.array([[0, 1, 4], [1, 2, 4], [2, 5, 4], [0, 4, 3], [3, 4, 6], [4, 7, 6], [8, 7, 4], [4 ,5, 8]]))
mesh1 = o3d.geometry.TriangleMesh(vertices1, triangles1)

vertices2 = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1.5, 0, 0], [2, 0, 0], [0, 0.5, 0], [1, 1.5, 0], [2, 1, 0], [0, 2, 0], [1.5, 2, 0], [2, 2, 0]]) +
        [3, 0, 0]
        )
triangles2 = o3d.utility.Vector3iVector(
        np.array([[0, 1, 4], [1, 2, 4], [2, 5, 4], [0, 4, 3], [3, 4, 6], [4, 7, 6], [8, 7, 4], [4 ,5, 8]]))
mesh2 = o3d.geometry.TriangleMesh(vertices2, triangles2)


#o3d.visualization.draw_geometries([mesh1, mesh2], mesh_show_wireframe=True) 


pointcloud1 = mesh1.sample_points_uniformly(number_of_points=2000)
pointcloud2 = mesh2.sample_points_uniformly(number_of_points=2000)
#o3d.visualization.draw([pointcloud1, pointcloud2], point_size=2, show_skybox=False, bg_color=(1.0, 1.0, 1.0, 1.0)) 

tensor_pointcloud1 = torch.from_numpy(np.asarray(pointcloud1.points)).view(1, 2000, 3).to(torch.float32)
tensor_pointcloud2 = torch.from_numpy(np.asarray(pointcloud2.points) - [3, 0, 0]).view(1, 2000, 3).to(torch.float32)

cd, _ = cuda_cd(tensor_pointcloud1, tensor_pointcloud2)

print(cd)