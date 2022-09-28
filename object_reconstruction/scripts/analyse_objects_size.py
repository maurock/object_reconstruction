import numpy as np
import os
from object_reconstruction.utils import mesh_utils
from glob import glob
import object_reconstruction.data.objects as objects
import object_reconstruction.data.obj_pointcloud as obj_pointcloud
import object_reconstruction.data.touch_charts as touch_charts
import trimesh
"""
Script to calculate the size of the objects in tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/objects
"""

def main():   
    # List all the objects in /data/obj_pointcloud/
    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(os.path.dirname(objects.__file__), '*'))]
    list_objects.remove('__init__.py')
    list_objects.remove('__pycache__')

    # Objects size
    for obj_index in list_objects:
        filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
        verts, faces = mesh_utils.mesh_from_urdf(filepath_obj)
        print(f'Object {obj_index}: \n Minimum x: {np.amin(verts[:, 0])}, Minimum y: {np.amin(verts[:, 1])}, Minimum z: {np.amin(verts[:, 2])} \n Maximum x: {np.amax(verts[:, 0])}, Maximum y: {np.amax(verts[:, 1])}, Maximum z: {np.amax(verts[:, 2])}')

    # Initial sphere size
    initial_sphere_path = os.path.join(os.path.dirname(touch_charts.__file__), 'vision_charts.obj')
    verts, faces = mesh_utils.load_mesh_touch(initial_sphere_path)
    verts = verts.numpy()
    print(f'Initial mesh: \n Minimum x: {np.amin(verts[:, 0])}, Minimum y: {np.amin(verts[:, 1])}, Minimum z: {np.amin(verts[:, 2])} \n Maximum x: {np.amax(verts[:, 0])}, Maximum y: {np.amax(verts[:, 1])}, Maximum z: {np.amax(verts[:, 2])}')    

if __name__=='__main__':
    main()