import numpy as np
import os
import object_reconstruction.data.touch_charts as touch_charts
from object_reconstruction.utils import mesh_utils
import plotly.graph_objects as go
import trimesh
import meshplot as mp
import matplotlib.pyplot as plt
import object_reconstruction.data.objects as objects
import object_reconstruction.data.obj_pointcloud as obj_pointcloud
import open3d as o3d

plot_mesh_object = False
plot_mesh_touch_chart = False
plot_initial_chart = False
plot_obj_pointcloud = True

if plot_mesh_object:
    # Plot object
    object_path = os.path.join(os.path.dirname(objects.__file__), '102763')
    object_mesh = mesh_utils.mesh_from_urdf(object_path)

    verts = object_mesh[0]
    faces = object_mesh[1]
    # Deform the object
    verts[:, 0] = verts[:, 0] + np.random.normal(0, 0.03, size=verts[:, 0].shape)
    verts[:, 1] = verts[:, 1] + np.random.normal(0, 0.03, size=verts[:, 0].shape)
    verts[:, 2] = verts[:, 2] + np.random.normal(0, 0.03, size=verts[:, 0].shape)

    # Open3D
    verts = o3d.utility.Vector3dVector(verts)
    faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(verts, faces)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True) 

    # Trimesh
    # scene = trimesh.scene.Scene(geometry=trimesh.Trimesh(object_mesh[0], object_mesh[1]))
    # scene.show()

elif plot_mesh_touch_chart:
    # Settings
    object_id = '102763'
    touch_charts_dict = np.load(os.path.join(os.path.dirname(touch_charts.__file__), object_id, 'touch_charts_gt.npy'), allow_pickle=True).item()
    
    # TO DO...

elif plot_initial_chart:
    # Settings
    object_id = 'vision_charts.obj'
    chart_path = os.path.join(os.path.dirname(touch_charts.__file__), object_id)
    initial_verts, initial_faces = mesh_utils.load_mesh_touch(chart_path)
    initial_verts = initial_verts.numpy()
    initial_faces = initial_faces.numpy()

    # Deform the chart
    # initial_verts[:, 0] = initial_verts[:, 0] + np.random.normal(0, 0.01, size=initial_verts[:, 0].shape)
    # initial_verts[:, 1] = initial_verts[:, 1] + np.random.normal(0, 0.01, size=initial_verts[:, 0].shape)
    # initial_verts[:, 2] = initial_verts[:, 2] + np.random.normal(0, 0.01, size=initial_verts[:, 0].shape)

    initial_verts = o3d.utility.Vector3dVector(initial_verts)
    initial_faces = o3d.utility.Vector3iVector(initial_faces)
    mesh = o3d.geometry.TriangleMesh(initial_verts, initial_faces)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True) 

elif plot_obj_pointcloud:
    object_id = '102763'
    obj_pointcloud = np.load(os.path.join(os.path.dirname(obj_pointcloud.__file__), object_id, 'obj_pointcloud.npy'), allow_pickle=True)  # (2000, 3)
    
    obj_pointcloud[:, 0] = obj_pointcloud[:, 0] + np.random.normal(0, 0.05, size=obj_pointcloud[:, 0].shape)
    obj_pointcloud[:, 1] = obj_pointcloud[:, 1] + np.random.normal(0, 0.05, size=obj_pointcloud[:, 0].shape)
    obj_pointcloud[:, 2] = obj_pointcloud[:, 2] + np.random.normal(0, 0.05, size=obj_pointcloud[:, 0].shape)

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    # Create plots
    x = obj_pointcloud[:, 0]
    y = obj_pointcloud[:, 1]
    z = obj_pointcloud[:, 2]
    ax.set_box_aspect([1,1,1])
    ax.set(xlim3d=(0, 1), xlabel='X')
    ax.set(ylim3d=(0, 1), ylabel='Y')
    ax.set(zlim3d=(0, 1), zlabel='Z')
    scatters = ax.scatter(x, y, z, s=20, c='blue')
    plt.axis('off')
    plt.grid(visible=None)
    ax.view_init(70, 60)
    plt.show()
