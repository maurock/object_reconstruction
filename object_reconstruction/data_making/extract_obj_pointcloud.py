import numpy as np
import os
from object_reconstruction.utils import mesh_utils
from glob import glob
import object_reconstruction.data.objects as objects
import object_reconstruction.data.obj_pointcloud as obj_pointcloud
import trimesh
import argparse
from copy import deepcopy
import pybullet as pb

"""
Script to extract pointclouds from objects in tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/objects.
The point cloud is scaled and rotated as the objects loaded in PyBullet.
"""

def save_obj_pointcloud(obj_index, pointcloud):
    dir_obj_pointcloud = os.path.join(os.path.dirname(obj_pointcloud.__file__), obj_index)
    if not os.path.exists(dir_obj_pointcloud):
        os.makedirs(dir_obj_pointcloud)       
        filepath = os.path.join(dir_obj_pointcloud, f'obj_pointcloud.npy')
        np.save(filepath, pointcloud)

def main(args):   
    # List all the objects in /data/obj_pointcloud/
    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(os.path.dirname(objects.__file__), '*'))]
    list_objects.remove('__init__.py')
    list_objects.remove('__pycache__')

    for obj_index in list_objects:
        filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
        verts, faces = mesh_utils.mesh_from_urdf(filepath_obj)
        pointcloud = mesh_utils.mesh_to_pointcloud(verts, faces, args.num_samples)
        # The object point cloud needs to be scaled and rotated, as it is scaled and rotated during data collection.
        pointcloud_s = mesh_utils.scale_pointcloud(pointcloud)
        pointcloud_s_r = mesh_utils.rotate_pointcloud(pointcloud_s)
        save_obj_pointcloud(obj_index, pointcloud_s_r)      

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="Number of samples for the full object pointcloud"
    )
    args = parser.parse_args()

    main(args)