import time
import numpy as np
import pybullet as p
from object_reconstruction.utils.shoot_rays_utils import *
from object_reconstruction.utils.obj_utils import *
import os
import object_reconstruction.data.touch_charts as touch_charts
import object_reconstruction.data.checkpoints as checkpoints
from object_reconstruction.utils.mesh_utils import *
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from object_reconstruction.utils.misc_utils import render_scene
from copy import deepcopy

def robot_touch_vertical(robot, sample, max_height_wrld=0.2):
    """Given x-y coordinates in the worldframe, the robot moves to a high position and then 
    goes down to sample the object's surface"""
    sample_high_wrld = [sample[0], sample[1], max_height_wrld]
    sample_high_wrk, _ = robot.arm.worldframe_to_workframe(sample_high_wrld, [0, 0, 0])
    robot.move_linear(sample_high_wrk, [0, 0, 0])
    time.sleep(0.1)
    robot.stop_at_touch = True
    sample_low_wrld = [sample[0], sample[1], max_height_wrld] + np.array([0, 0, -max_height_wrld])
    sample_low_wrk, _ = robot.arm.worldframe_to_workframe(sample_low_wrld, [0, 0, 0])
    robot.move_linear(sample_low_wrk , [0, 0, 0])
    time.sleep(0.1)
    robot.stop_at_touch = False
    return robot

def sample_hemisphere(r):
    """
    Uniform sampling on a hemisphere.
    Parameter:
        - r: radius
    Returns:
        - [x, y, z]: list of points in world frame
        - [phi, theta]: phi is horizontal (0, pi/2), theta is vertical (0, pi/2) 
    """
    phi = 2 * np.pi * np.random.uniform()
    theta = np.arccos(1 - np.random.uniform())
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = np.absolute(r * np.cos(theta))
    coords = [x, y, z]
    angles = [phi, theta]
    return coords, angles

def _debug_plot_sphere(r, origin):
    """ 
    Draw a sphere (made of points) around the object
    """
    coords_array = np.array([]).reshape(0, 3)
    for i in range(0, 1000):
        coords, _ = sample_hemisphere(r)
        coords = coords + np.array(origin)
        coords_array = np.vstack((coords_array, coords))
    color_array = np.full(shape=coords_array.shape, fill_value=np.array([0, 72, 255])/255)
    p.addUserDebugPoints(
            pointPositions=coords_array, 
            pointColorsRGB=color_array,
            pointSize=2)

def _debug_pointcloud_to_mesh(mesh, robot, rot_M_wrld, filtered_full_pointcloud, debug_pointcloud_to_mesh, obj_index, iteration):
    mesh_array = np.asarray(mesh.vertices)[np.newaxis, :, :]
    mesh_wrld = translate_rotate_mesh(np.array([robot.coords_at_touch_wrld]), rot_M_wrld, 
                                        mesh_array, obj_initial_pos=[0, 0, 0])
    debug_pointcloud_to_mesh[obj_index][iteration] = [filtered_full_pointcloud, mesh_wrld]  
    debug_pointcloud_to_mesh_dir = os.path.join(os.path.dirname(checkpoints.__file__), 'debug_pointcloud_to_mesh')
    debug_pointcloud_to_mesh_path = os.path.join(debug_pointcloud_to_mesh_dir, 'debug_pointcloud_to_mesh.npy')
    np.save(debug_pointcloud_to_mesh_path, debug_pointcloud_to_mesh)

def _debug_rotation(filtered_full_pointcloud, pointcloud_wrk, robot, pos_wrk, rot_Q_wrld, rot_M_wrld, initial_pos, obj_index, debug_rotation):
    dict_new = deepcopy(debug_rotation)
    if len(dict_new[obj_index].keys()) == 0:
        dict_new[obj_index] = dict({
            'filtered_full_pointcloud': [],
            'pointcloud_wrk': [],
            'pos_wrld': [],
            'pos_wrk': [],
            'rot_Q_wrld': [],
            'rot_M_wrld': [],
            'initial_pos': []
        })
    timestamp_run = datetime.now().strftime('%d_%m_%H%M') 
    rotation_dir = os.path.join(os.path.dirname(checkpoints.__file__), f'debug_rotation')
    dict_new[obj_index]['filtered_full_pointcloud'].append(filtered_full_pointcloud)
    dict_new[obj_index]['pointcloud_wrk'].append(pointcloud_wrk)
    dict_new[obj_index]['pos_wrld'].append(robot.coords_at_touch_wrld)
    dict_new[obj_index]['pos_wrk'].append(pos_wrk)
    dict_new[obj_index]['rot_Q_wrld'].append(rot_Q_wrld)
    dict_new[obj_index]['rot_M_wrld'].append(rot_M_wrld)
    dict_new[obj_index]['initial_pos'].append(initial_pos)
    np.save(os.path.join(rotation_dir, f'debug_rotation_{timestamp_run}.npy'), dict_new)
    return dict_new

def sphere_orn_wrld(robot, origin, angles):
    """
    Params:
        - angles: list [phi, theta], where phi is horizontal, theta is vertical
    The rotation matrix was taken from https://en.wikipedia.org/wiki/Spherical_coordinate_system
    and modified by swapping the x and z axes, and negating the z axis so that the robot goes towards the object.
    """
    phi, theta = angles
    rot_M = np.array([  [-np.sin(phi), np.cos(theta) * np.cos(phi), -np.sin(theta)*np.cos(phi)],
                        [np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta) * np.sin(phi)],
                        [0, -np.sin(theta), -np.cos(theta)]])
    pb.addUserDebugLine(origin, (origin + rot_M[:, 0]), lineColorRGB=[1,0,0], lifeTime=20)
    pb.addUserDebugLine(origin, (origin + rot_M[:, 1]), lineColorRGB=[0,1,0], lifeTime=20)
    pb.addUserDebugLine(origin, (origin + rot_M[:, 2]), lineColorRGB=[0,0,1], lifeTime=20)
    r = R.from_matrix(rot_M)
    orn = r.as_euler('xyz')
    return orn

def move_wrld_to_work(robot, pos_wrld, orn_wrld=[3.14, 0, 1.57]):
    pos_wrk, orn_wrk = robot.arm.worldframe_to_workframe(pos_wrld, orn_wrld)
    robot.move_linear(pos_wrk, orn_wrk)
    time.sleep(0.1)

def robot_touch_spherical(robot, robot_sphere_wrld, initial_pos, angles, max_height_wrld=0.2):
    """Given x-y coordinates in the worldframe, the robot moves to a high position and then goes down to sample the object's surface"""
    # go to high location
    high_wrld = np.array(initial_pos) + np.array([0, 0, max_height_wrld])
    move_wrld_to_work(robot, high_wrld)
    # go to pos on sphere and randomly move the end effector
    orn_wrld = sphere_orn_wrld(robot, robot_sphere_wrld, angles)
    move_wrld_to_work(robot, robot_sphere_wrld, orn_wrld)
    robot.stop_at_touch = True
    move_wrld_to_work(robot, initial_pos, orn_wrld)    # This does not work in DIRECT mode
    robot.stop_at_touch = False
    return robot

def spherical_sampling(robot, obj_id, initial_pos, initial_orn, args, obj_index, num_points=2000, scale=1, nx=5, ny=5, mesh=None, obj=False):
    """
    This method computes the max and min coordinates and samples randomly within these boundaries.
    If the mesh is loaded as .urdf, we use pybullet.getMesh(). If it is a .obj, we pass a Trimesh object and extracts the vertices.

    Return:
        - samples_list = np.array containing list of full pointclouds at touch (variable number of points)
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
    """
    if obj:
        vertices_wrld = mesh.vertices
        vertices_wrld = np.array(vertices_wrld) * scale + initial_pos   # add initial obj position and scale
    else:
        # worldframe coordinates. These do not take into account the initial obj position.
        num_vertices, vertices_wrld = p.getMeshData(obj_id, 0)   
        # show object vertices
        initial_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
        vertices_wrld = rotate_vector_by_quaternion(np.array(vertices_wrld), initial_orn) + initial_pos

    # Get min and max world object coordinates. 
    min_coords = [ np.amin(vertices_wrld[:,0]), np.amin(vertices_wrld[:,1]), np.amin(vertices_wrld[:,2]) ]
    max_coords = [ np.amax(vertices_wrld[:,0]), np.amax(vertices_wrld[:,1]), np.amax(vertices_wrld[:,2]) ]

    print(f'Minimum: {min_coords[0]}, {min_coords[1]}, {min_coords[2]}')        
    print(f'Maximum: {max_coords[0]}, {max_coords[1]}, {max_coords[2]}')
    
    # ray: sqrt( (x1 - xc)**2 + (y1 - yc)**2)
    ray_hemisphere = 1.5 * np.sqrt((max_coords[0] - initial_pos[0])**2 + (max_coords[1] - initial_pos[1])**2 + (max_coords[2] - initial_pos[2])**2)
    
    # Set pointcloud grid
    robot.nx = nx
    robot.ny = ny

    # Initialize lists
    pos_wrld_list = np.array([], dtype=np.float32).reshape(0, 3)  # TCP pos (worldframe)
    pos_wrk_list = np.array([], dtype=np.float32).reshape(0, 3)   # TCP pos (worldframe)
    mesh_list = []     # touch chart containing 25 vertices (workframe)
    tactile_imgs = np.array([], dtype=np.float32).reshape(0, 256, 256)
    pointcloud_list = np.array([], dtype=np.float32).reshape(0, 2000, 3)   # fixed dimension touch chart pointcloud (workframe)
    rot_M_wrld_list = np.array([], dtype=np.float32).reshape(0, 3, 3)      # rotation matrix (work wrt worldframe)
    
    # Initialize lists for debug
    debug_pointcloud_to_mesh = dict()    # debug pointcloud_to_mesh
    debug_pointcloud_to_mesh[obj_index] = dict()
    debug_rotation = dict()
    debug_rotation[obj_index] = dict()
    for iteration in range(args.num_samples):
        robot.results_at_touch_wrld = None
        hemisphere_random_pos, angles = sample_hemisphere(ray_hemisphere)
        #_debug_plot_sphere(ray_hemisphere, initial_pos)
        robot_sphere_wrld = np.array(initial_pos) + np.array(hemisphere_random_pos)
        robot_touch_spherical(robot, robot_sphere_wrld, initial_pos, angles)

        if args.render_scene:
            render_scene()

        # Show mesh obtained from pointcloud using Open3D.
        if args.debug_show_full_mesh:
            pointcloud_to_mesh(robot.results_at_touch_wrld[:, 3], args)

        # Plot full pointcloud using Plotly
        if args.debug_contact_points:
            _debug_contact_points(robot, obj_id)

        # If the robot touches the object, get mesh from pointcloud using Open3D, optionally visualise it. If not contact points, continue. 
        if robot.results_at_touch_wrld is None:
            continue
        filtered_full_pointcloud = filter_point_cloud(robot.results_at_touch_wrld)
        if filtered_full_pointcloud.shape[0] < 26:
            print('Point cloud shape is too small')
            continue

        # Full pointcloud to 25 vertices. By default, vertices are converted to workframe.
        mesh = pointcloud_to_vertices_wrk(filtered_full_pointcloud, robot, args)
        if (np.asarray(mesh.vertices).shape[0] != 25) or (np.asarray(mesh.triangles).shape[0] == 0):
            print('Mesh does not have 25 vertices or faces not found')
            continue
        mesh_list.append(mesh)

        # Store world position of the TCP
        pos_wrld_list = np.vstack((pos_wrld_list, robot.coords_at_touch_wrld))

        # Store tactile images
        camera = robot.get_tactile_observation()[np.newaxis, :, :]
        tactile_imgs = np.vstack((tactile_imgs, camera))

        pointcloud_wrk = mesh.sample_points_poisson_disk(num_points).points  #PointCloud obj wrk
        pointcloud_wrk = np.array(pointcloud_wrk, dtype=np.float32)[None, :, :] # higher dimension for stacking
        pointcloud_list = np.vstack((pointcloud_list, pointcloud_wrk))

        # Store pose and rotation
        pos_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[0]
        rot_Q_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
        rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        pos_wrk_list = np.vstack((pos_wrk_list, pos_wrk))
        rot_M_wrld_list = np.vstack((rot_M_wrld_list, rot_M_wrld))

        # Store mesh and point cloud and check if their position matches
        if args.debug_pointcloud_to_mesh:
            _debug_pointcloud_to_mesh(mesh, robot, rot_M_wrld, filtered_full_pointcloud, debug_pointcloud_to_mesh, obj_index, iteration)
            
        # debug position and rotation
        if args.debug_rotation:
            debug_rotation = _debug_rotation(
                filtered_full_pointcloud, pointcloud_wrk, robot, pos_wrk, rot_Q_wrld, 
                rot_M_wrld, initial_pos, obj_index, debug_rotation
            )

        # intermediate saving
        save_touch_charts(mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list, initial_pos)

    return mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list

def vertical_sampling(robot, obj_id, initial_pos, initial_orn, args, obj_index, num_points=2000, scale=1, nx=5, ny=5, mesh=None, obj=False):
    """
    This method computes the max and min coordinates and samples randomly within these boundaries.
    If the mesh is loaded as .urdf, we use pybullet.getMesh(). If it is a .obj, we pass a Trimesh object and extracts the vertices.

    Return:
        - samples_list = np.array containing list of full pointclouds at touch (variable number of points)
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
    """
    if obj:
        vertices_wrld = mesh.vertices
        vertices_wrld = np.array(vertices_wrld) * scale + initial_pos   # add initial obj position and scale
    else:
        # worldframe coordinates. These do not take into account the initial obj position.
        num_vertices, vertices_wrld = p.getMeshData(obj_id, 0)   
        # show object vertices
        initial_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
        vertices_wrld = rotate_vector_by_quaternion(np.array(vertices_wrld), initial_orn) + initial_pos

    # Snippet to get min and max world object coordinates. 
    print(f'Minimum: {np.amin(vertices_wrld[:,0]), np.amin(vertices_wrld[:,1])}')        
    print(f'Maximum: {np.amax(vertices_wrld[:,0]), np.amax(vertices_wrld[:,1])}')     
    
    # set boundaries
    x_lim_wrld = [np.amin(vertices_wrld[:,0]), np.amax(vertices_wrld[:,0])] 
    y_lim_wrld = [np.amin(vertices_wrld[:,1]), np.amax(vertices_wrld[:,1])]

    # Set pointcloud grid
    robot.nx = nx
    robot.ny = ny

    # Initialize lists
    pos_wrld_list = np.array([], dtype=np.float32).reshape(0, 3)
    mesh_list = []
    tactile_imgs = np.array([], dtype=np.float32).reshape(0, 256, 256)
    pointcloud_list = np.array([], dtype=np.float32).reshape(0, 2000, 3)   # pointcloud_wrk
    rot_M_wrld_list = np.array([], dtype=np.float32).reshape(0, 3, 3)
    pos_wrk_list = np.array([], dtype=np.float32).reshape(0, 3)
    initial_pos_list = np.array([], dtype=np.float32).reshape(0, 3)
    for _ in range(args.num_samples):
        sample = np.random.uniform( low = [x_lim_wrld[0], y_lim_wrld[0]], 
                                    high = [x_lim_wrld[1], y_lim_wrld[1]],
                                    size = 2)   # x-y coords of random sampling
        print(f'Random sample: {sample}')
        robot_touch(robot, sample)

        # Show mesh obtained from pointcloud using Open3D.
        if args.debug_show_full_mesh:
            pointcloud_to_mesh(robot.results_at_touch_wrld[:, 3], args)

        # Plot full pointcloud using Plotly
        if args.debug_contact_points:
            _debug_contact_points(robot, obj_id)

        # If the robot touches the object, get mesh from pointcloud using Open3D, optionally visualise it. If not contact points, continue. 
        if robot.results_at_touch_wrld is None:
            continue
        filtered_full_pointcloud = filter_point_cloud(robot.results_at_touch_wrld)
        if filtered_full_pointcloud.shape[0] < 26:
            print('Point cloud shape is too small')
            continue
        
        # Full pointcloud to 25 vertices. By default, vertices are converted to workframe.
        mesh  = pointcloud_to_vertices_wrk(filtered_full_pointcloud, robot, args)
        if np.asarray(mesh.vertices).shape[0] != 25:
            print('Mesh does not have 25 vertices')
            continue
        mesh_list.append(mesh)

        # Store full pointcloud in world frame (containing a variable number of contact points)
        pos_wrld_list = np.vstack((pos_wrld_list, robot.coords_at_touch_wrld))

        # Store tactile images
        camera = robot.get_tactile_observation()[np.newaxis, :, :]
        tactile_imgs = np.vstack((tactile_imgs, camera))

        # Vertices and faces to point cloud if mesh contains triangles (sometimes it does not)
        if np.asarray(mesh.triangles).shape[0] == 0:
            continue
        pointcloud_wrk = mesh.sample_points_poisson_disk(num_points).points  #PointCloud obj wrk
        pointcloud_wrk = np.array(pointcloud_wrk, dtype=np.float32)[None, :, :] # higher dimension for stacking
        pointcloud_list = np.vstack((pointcloud_list, pointcloud_wrk))

        # Store pose and rotation
        pos_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[0]
        rot_Q_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
        rot_M_wrld = np.array(pb.getMatrixFromQuaternion(rot_Q_wrld)).reshape(1, 3, 3)
        pos_wrk_list = np.vstack((pos_wrk_list, pos_wrk))
        rot_M_wrld_list = np.vstack((rot_M_wrld_list, rot_M_wrld))

        # Store initial position
        initial_pos_list = np.vstack((initial_pos_list, initial_pos))

        # intermediate saving
        save_touch_charts(mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list)

    return mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list

def save_touch_charts(mesh_list, tactile_imgs, pointcloud_list, obj_index, rot_M_wrld_list, pos_wrld_list, pos_wrk_list, initial_pos):
    """
    Receive list containing open3D.TriangleMesh of the local touch charts (25 vertices) and tactile images related to those meshes. It saves a dictionary containing vertices and faces as np array, and normalised tactile images. 

    Parameters:
        - mesh_list = list containing open3d.geometry.TriangleMesh (25 vertices and faces of the local geometry at touch site)
        - tactile_imgs = list of tactile images, np.array(1, 256, 256)
        - pointcloud_list = list of pointclouds, containing 2000 randomly sampled points that    represent the ground truth to compute the chamfer distance
        - obj_index: index of the object, e.g. camera: 101352
        - rot_M_wrld_list: list of rotation matrices to convert from workframe to worldframe. np.array, shape (n, 3, 3)
        - pos_wrld_list: list of positions of the TCP in worldframe. np.array, shape(n, 3)
        - pos_wrk_list: list of positions of the TCP in workframe. np.array, shape(n, 3)
    Returns:
        - touch_charts_data, dictionary with keys: 'verts', 'faces', 'tactile_imgs', 'pointclouds', 'rot_M_wrld;, 'pos_wrld', 'pos_wrk'
            - 'verts': shape (n_samples, 75), ground truth vertices for various samples
            - 'faces': shape (n_faces, 3), concatenated triangles. The number of faces per sample varies, so it is not possible to store faces per sample.
            - 'tactile_imgs': shape (n_samples, 1, 256, 256)
            - 'pointclouds': shape (n_samples, 2000, 3), points randomly samples on the touch charts mesh surface.
            - 'rot_M_wrld': 3x3 rotation matrix collected from PyBullet.
            - 'pos_wrld': position of the sensor in world coordinates at touch, collected from PyBullet (robots.coords_at_touch)
            - 'pos_wrk': position of the sensor in world frame collected from PyBullet.
    """
    verts = np.array([], dtype=np.float32).reshape(0, 75)
    faces = np.array([], dtype=np.float32).reshape(0, 3)
    touch_charts_data = dict()
    for mesh in mesh_list:
        vert = np.asarray(mesh.vertices, dtype=np.float32).ravel()
        verts = np.vstack((verts, vert))
        faces = np.vstack((faces, np.asarray(mesh.triangles, dtype=np.float32)))   # (n, 3) not possible (b, n, 3) because n is not constant
    touch_charts_data['verts'] = verts
    touch_charts_data['faces'] = faces

    # Conv2D requires [batch, channels, size1, size2] as input. tactile_imgs is currently [num_samples, size1, size2]. I need to add a second dimension.
    tactile_imgs = np.expand_dims(tactile_imgs, 1) / 255     # normalize tactile images
    touch_charts_data['tactile_imgs'] = tactile_imgs
    touch_charts_data['pointclouds'] = pointcloud_list

    # Store data for rotation and translation
    touch_charts_data['rot_M_wrld'] = rot_M_wrld_list
    touch_charts_data['pos_wrld'] = pos_wrld_list
    touch_charts_data['pos_wrk'] = pos_wrk_list
    touch_charts_data['initial_pos'] = initial_pos

    touch_charts_data_path = os.path.join(os.path.dirname(touch_charts.__file__), obj_index)

    if not os.path.isdir(touch_charts_data_path):
        os.mkdir(touch_charts_data_path)
    touch_charts_data_path = os.path.join(touch_charts_data_path, 'touch_charts_gt.npy')

    np.save(touch_charts_data_path, touch_charts_data)
