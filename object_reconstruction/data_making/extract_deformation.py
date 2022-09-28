import numpy as np
import argparse
from glob import glob
import os
import object_reconstruction.data.touch_charts as touch_charts
import object_reconstruction.data.objects as objects
import object_reconstruction.data.obj_pointcloud as obj_pointcloud
from object_reconstruction.utils.mesh_utils import *
import random
from copy import deepcopy
"""
Take the touch charts created by extract_touch_charts.npy and randomly combine them with vision charts and empty charts. Store object pointclouds.
"""

def generate_touch_vision_data(
    init_touch_faces,
    init_vision_verts,
    init_vision_faces,
    touch_charts_dict,
    max_touches, 
    num_datapoints):
    """
    Params:
        - init_touch_faces: faces of initial mesh sheet, torch.Tensor()
        - init_vision_verts: verts of initial spherical mesh, torch.Tensor shape (1824, 3)
        - init_vision_faces: faces of initial spherical mesh, torch.Tensor(), shape (2304, 3)
        - touch_charts_dict: dictionary containing four elements: 'verts', 'faces', 'tactile_imgs', 'pointclouds' 
        - max_touches: number of max touches allowed
        - num_datapoints: number of datapoints to generate
    Returns:
        - touch_vision_verts: combination of verts for initial sphere and touch charts (both the randomly sampled ones and empty vertices), shape (args.num_datapoints, num_faces, 3), for 5 max_touches and 10 datapoints: shape (10, 1949, 3)
        - touch_vision_faces: combination of verts for initial sphere and predetermined faces for the touch charts, shape (args.max_touches, num_verts, 3), for 5 max_touches and 10 datapoints: shape (10, 2464, 3)
        - ad_info: dictionary containing keys 'original', 'adj', 'faces'
            - 'original': normalised adjacency matrix for initial sphere, shape(1824, 1824) 
            - 'adj': normalised adjacency matrix of initial sphere + touch charts
            - 'faces': faces of initial sphere + touch chart
    """     
    # num faces for initial sphere (2304) + args.max_touches * faces in each touch chart (32)
    num_faces = 2304 + args.max_touches * 32
    # num verts for initial sphere (1824) + args.max_touches * verts in each touch chart (25)
    num_verts = 1824 + args.max_touches * 25    
    # initialize np arrays for collecting data
    touch_vision_verts = np.array([], dtype=np.float32).reshape(0, num_verts, 3)     
    touch_vision_faces = np.array([], dtype=np.int64).reshape(0, num_faces, 3) 
    touch_vision_mask = np.array([], dtype=np.float32).reshape(0, num_verts)
    for _ in range(0, num_datapoints):
        num_touches = random.sample(list(np.arange(1, max_touches + 1, 1, dtype=int)), 1)[0]   # num between 1 and max_touches+1 
        verts, faces, mask = _combine_touch_vision_charts(touch_charts_dict, init_touch_faces, init_vision_verts, init_vision_faces, num_touches, max_touches)
        touch_vision_verts = np.concatenate((touch_vision_verts, verts[None, :, :]))
        touch_vision_faces = np.concatenate((touch_vision_faces, faces[None, :, :]))
        touch_vision_mask = np.concatenate((touch_vision_mask, mask[None, :]))
    # Adjacency matrix
    adj_info = adjacency_matrix(touch_vision_faces[0].astype(np.int32), touch_vision_verts[0], max_touches)
    return touch_vision_verts, touch_vision_faces, touch_vision_mask, adj_info

def _combine_touch_vision_charts(touch_charts_dict, init_touch_faces, init_vision_verts, init_vision_faces, num_touches, max_touches):
    """
    Params:
        - touch_charts_dict: dictionary containing four elements: 'verts', 'faces', 'tactile_imgs', 'pointclouds'
            # - 'verts': shape (n_samples, 75), ground truth vertices for various samples
            # - 'faces': shape (n_faces, 3), concatenated triangles. The number of faces per sample varies, so it is not possible to store faces per sample. ------ ? NOT SURE
            # - 'tactile_imgs': shape (n_samples, 1, 256, 256)
            # - 'pointclouds': shape (n_samples, 2000, 3), points randomly samples on the touch charts mesh surface.
            - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
            - tactile_imgs = list of tactile images, np.array(1, 256, 256)
            - pointcloud_list = list of pointclouds, containing 2000 randomly sampled points that    represent the ground truth to compute the chamfer distance
            - obj_index: index of the object, e.g. camera: 101352
            - rot_M_wrld_list: list of rotation matrices to convert from workframe to worldframe. np.array, shape (n, 3, 3)
            - pos_wrld_list: list of positions of the TCP in worldframe. np.array, shape(n, 3)
            - pos_wrk_list: list of positions of the TCP in workframe. np.array, shape(n, 3)
        - vision_charts_path: path to obj containing initial spherical mesh, shape (1824, 3)

    Returns:
        - verts: combination of verts for initial sphere and touch charts (both the randomly sampled ones and empty vertices), shape (args.num_datapoints, num_faces, 3), e.g. for 5 max_touches: shape (1949, 3)
        - faces: combination of verts for initial sphere and predetermined faces for the touch charts, shape (args.max_touches, num_verts, 3), e.g. for 5 max_touches: shape (2464, 3)
        - mask: information about the type of vertex: 0: empty, 1: touch_chart, 2: initial. shape ( , )
    """ 
    num_touch_charts = touch_charts_dict['verts'].shape[0]

    faces = np.array([], dtype=np.float32).reshape(0, 3)
    verts = np.array([], dtype=np.float32).reshape(0, 3)

    # Add initial vision chart to verts and faces:
    # - verts
    index_touch_charts = np.random.choice(np.arange(0, num_touch_charts), size=num_touches)
    # translate and rotate vertices to global frame
    verts_wrld = translate_rotate_mesh(
        touch_charts_dict['pos_wrld'], touch_charts_dict['rot_M_wrld'], touch_charts_dict['verts'].reshape(-1, 25, 3), touch_charts_dict['initial_pos'])
    random_verts = verts_wrld[index_touch_charts, :, :]
    random_verts = random_verts.reshape(random_verts.shape[0] * random_verts.shape[1], 3)
    empty_verts = np.zeros(((max_touches - num_touches) * 25, 3))        
    verts = np.concatenate((verts, init_vision_verts.cpu().numpy()))
    verts = np.concatenate((verts, random_verts))
    verts = np.concatenate((verts, empty_verts)).astype(np.float32)

    # - Create mask (verts): 0: empty, 1: touch_chart, 2: initial
    mask_init = np.full(init_vision_verts.shape[0], 2)
    mask_touch = np.full(random_verts.shape[0], 1)
    mask_empty = np.full(empty_verts.shape[0], 0)
    mask = np.concatenate((mask_init, mask_touch, mask_empty), dtype=np.float32)

    # - faces
    faces = np.concatenate((faces, init_vision_faces.cpu()))
    #faces = np.concatenate((faces, np.tile(init_touch_faces, (max_touches, 1)))).astype(np.float32)
    for i in range(max_touches):
        faces_offset = init_touch_faces.clone() + init_vision_verts.shape[0]   # Offset faces: [[1824, 1825, 1827], [...]], shape [32, 3]
        faces_offset += i * 25   # offset by number of datapoint, there are 25 vertices per touch chart
        faces = np.concatenate((faces, faces_offset.cpu()))
    faces = faces.astype(np.int64)

    return verts, faces, mask

def main(args):
    # Set paths
    initial_vision_charts_path = os.path.join(os.path.dirname(touch_charts.__file__), 'vision_charts.obj')
    initial_touch_charts_path = os.path.join(os.path.dirname(touch_charts.__file__), 'touch_chart.obj')
    adj_info_path = os.path.join(os.path.dirname(touch_charts.__file__), 'adj_info.npy')

    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(os.path.dirname(touch_charts.__file__), '*'))]
    list_objects.remove('__init__.py')
    list_objects.remove('__pycache__')
    list_objects.remove('adj_info.npy')
    list_objects.remove('touch_chart.obj')
    list_objects.remove('vision_charts.obj')
    for obj_index in list_objects:
        touch_charts_path = os.path.join(os.path.dirname(touch_charts.__file__), obj_index, 'touch_charts_gt.npy')
        dict_deformation_path = os.path.join(os.path.dirname(touch_charts.__file__), obj_index, 'touch_vision.npy')
        #obj_pointcloud_path = os.path.join(os.path.dirname(obj_pointcloud.__file__), obj_index, 'obj_pointcloud.npy')

        # load touch charts and initial spherical mesh
        touch_charts_dict = np.load(touch_charts_path, allow_pickle=True).item()
        init_vision_verts, init_vision_faces = load_mesh_touch(initial_vision_charts_path)   # function returns tensors, not np.arrays
        _, init_touch_faces = load_mesh_touch(initial_touch_charts_path) # function returns tensors, not np.arrays

        touch_vision_verts, touch_vision_faces, touch_vision_mask, adj_info = generate_touch_vision_data(
            init_touch_faces,
            init_vision_verts,
            init_vision_faces,
            touch_charts_dict,
            args.max_touches, 
            args.num_datapoints)

        dict_deformation = dict()
        dict_deformation['verts'] = touch_vision_verts
        dict_deformation['faces'] = touch_vision_faces
        dict_deformation['mask'] = touch_vision_mask
        #dict_deformation['obj_pointcloud'] = np.load(obj_pointcloud_path, allow_pickle=True)

        np.save(dict_deformation_path, dict_deformation)
        np.save(adj_info_path, adj_info)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_touches", type=int, default=5, help=""
    )
    parser.add_argument(
        "--num_datapoints", type=int, default=100, help=""
    )
    args = parser.parse_args()

    main(args)