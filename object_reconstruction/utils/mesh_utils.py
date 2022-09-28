import numpy as np
import torch 
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
from pytorch3d.loss import chamfer_distance as cuda_cd
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
import torch.nn.functional as F
from pytorch3d.io.obj_io import load_obj
from glob import glob
import os
import trimesh
import plotly.graph_objects as go
import object_reconstruction.data.touch_charts as touch_charts
from copy import deepcopy
import object_reconstruction.data.objects as objects
import pybullet as pb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# returns the chamfer distance between a mesh and a point cloud (Ed. Smith)
def chamfer_distance(verts, faces, gt_points, num=1000, repeat=1):
    pred_points= batch_sample(verts, faces, num=num)
    cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
    if repeat > 1:
        cds = [cd]
        for i in range(repeat - 1):
            pred_points = batch_sample(verts, faces, num=num)
            cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
            cds.append(cd)
        cds = torch.stack(cds)
        cd = cds.mean(dim=0)
    return cd

 # sample points from a batch of meshes
def batch_sample(verts, faces, num=10000):
    # Pytorch3D based code
    bs = verts.shape[0]
    face_dim = faces.shape[0]
    vert_dim = verts.shape[1]
    # following pytorch3D convention shift faces to correctly index flatten vertices
    F = faces.unsqueeze(0).repeat(bs, 1, 1)
    F += vert_dim * torch.arange(0, bs).unsqueeze(-1).unsqueeze(-1).to(F.device)
    # flatten vertices and faces
    F = F.reshape(-1, 3)
    V = verts.reshape(-1, 3)
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(V, F)
        Ar = areas.reshape(bs, -1)
        Ar[Ar != Ar] = 0
        Ar = torch.abs(Ar / Ar.sum(1).unsqueeze(1))
        Ar[Ar != Ar] = 1

        sample_face_idxs = Ar.multinomial(num, replacement=True)
        sample_face_idxs += face_dim * torch.arange(0, bs).unsqueeze(-1).to(Ar.device)

    # Get the vertex coordinates of the sampled faces.
    face_verts = V[F]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(bs, num, V.dtype, V.device)

    # Use the barycentric coords to get a point on each sampled face.
    A = v0[sample_face_idxs]  # (N, num_samples, 3)
    B = v1[sample_face_idxs]
    C = v2[sample_face_idxs]
    samples = w0[:, :, None] * A + w1[:, :, None] * B + w2[:, :, None] * C

    return samples

def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimes.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def mesh_from_urdf(filepath):
    """
    Receives path to object index containing the .URDF and verts and faces (both np.array).
    Directory tree:
    - obj_idx
    |   - textured_objs
    |   |   - ...obj
    |- ...
    """
    total_objs = glob(os.path.join(filepath, 'textured_objs/*.obj'))
    verts = np.array([]).reshape((0,3))
    faces = np.array([]).reshape((0,3))

    mesh_list = []
    for obj_file in total_objs:
        mesh = _as_mesh(trimesh.load(obj_file))
        mesh_list.append(mesh)           
                
    verts_list = [mesh.vertices for mesh in mesh_list]
    faces_list = [mesh.faces for mesh in mesh_list]
    faces_offset = np.cumsum([v.shape[0] for v in verts_list], dtype=np.float32)   # num of faces per mesh
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]            # compute offset for faces, otherwise they all start from 0
    verts = np.vstack(verts_list).astype(np.float32)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)]).astype(np.float32)

    return verts, faces

def scale_pointcloud(pointcloud, scale=0.1):
    obj = deepcopy(pointcloud)
    obj = obj * scale
    return obj

def rotate_pointcloud(pointcloud, rot=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection
    """
    obj = deepcopy(pointcloud)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    obj = np.einsum('ij,kj->ik', rot_M_obj, obj).transpose(1, 0)
    return obj

def mesh_to_pointcloud(verts, faces, n_samples):
    """
    This method samples n points on a mesh. The number of samples for each face is weighted by its size. 

    Params:
        verts = vertices, np.array(n, 3)
        faces = faces, np.array(m, 3)
        n_samples: number of total samples
    
    Returns:
        pointcloud
    """
    mesh = trimesh.Trimesh(verts, faces)
    pointcloud, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pointcloud = pointcloud.astype(np.float32)
    return pointcloud

# loads the initial mesh and returns vertex, and face information as torch tensors (Ed. Smith)
def load_mesh_touch(obj):
    obj_info = load_obj(obj)
    verts = obj_info[0]
    faces = obj_info[1].verts_idx
    verts = torch.FloatTensor(verts)
    faces = torch.LongTensor(faces)
    return verts, faces
    
def debug_pointcloud_obj(pointcloud):
    # show pointcloud sampled from object
    # 3D plot (Plotly), except the points ([0, 0, 0])
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    fig = go.Figure(
        [go.Scatter3d(x=x, y=y,z=z, mode='markers',
            marker=dict(
                size=2
                )
        )]
    )
    fig.show()

# computes adjacency matrix from face information
def calc_adj(faces):
    v1 = faces[:, 0]
    v2 = faces[:, 1]
    v3 = faces[:, 2]
    num_verts = int(faces.max())
    adj = np.eye(num_verts + 1)

    adj[(v1, v2)] = 1
    adj[(v1, v3)] = 1
    adj[(v2, v1)] = 1
    adj[(v2, v3)] = 1
    adj[(v3, v1)] = 1
    adj[(v3, v2)] = 1

    return adj

# normalizes symetric, binary adj matrix such that sum of each row is 1
def normalize_adj(adj_matrix):
    """
    Normalise the adjacency matrix over rows. It receives a numpy array and returns a numpy array.
    """
    # convert to torch to use this function developed by Meta AI and then reconvert to numpy
    adj_matrix_tensor = torch.from_numpy(adj_matrix)
    rowsum = adj_matrix_tensor.sum(1)
    r_inv = (1. / rowsum).view(-1)
    r_inv[r_inv != r_inv] = 0.
    adj_matrix_tensor = torch.mm(torch.eye(r_inv.shape[0]).to(adj_matrix_tensor.device) * r_inv, adj_matrix_tensor)
    adj_matrix = adj_matrix_tensor.numpy().astype(np.float32)
    return adj_matrix

# combines graph for vision and touch charts to define a fused adjacency matrix
def adj_fuse_touch(verts, adj_spherical_init, max_touches):
    """
    Parameters:
        - verts: initial spherical mesh + touch charts
        - adj_spherical_init: adjacency matrix of the initial spherical mesh
        - max_touches: maximum touches allowed, in the Meta AI paper they are 5.
    """
    verts_init = verts[:1824, :]   # 1824 vertices in initial sphere
    hash = {}
    # find vertices which have the same 3D position
    for e, v in enumerate(verts_init):
        if v.tobytes() in hash:
            hash[v.tobytes()].append(e)
        else:
            hash[v.tobytes()] = [e]

    # load object information for initial touch chart
    touch_chart_init_path = os.path.join(os.path.dirname(touch_charts.__file__), "touch_chart.obj")
    touch_chart_init_verts, touch_chart_init_faces = load_mesh_touch(touch_chart_init_path)
    touch_chart_init_adj = calc_adj(touch_chart_init_faces)

    # central vertex for each touch chart that will communicate with all vision charts
    central_point = 4
    central_points = [central_point + (i * touch_chart_init_adj.shape[0]) + adj_spherical_init.shape[0] for i in
                        range(max_touches)]   # [1824+4, 1824+25+4, ...] for <max_touches> times

    # define and fill new adjacency matrix with vision and touch charts
    new_adj = np.zeros((verts.shape[0], verts.shape[0]))  # verts is both spherical mesh and all the touch charts
    new_adj[:adj_spherical_init.shape[0], :adj_spherical_init.shape[0]] = deepcopy(adj_spherical_init)
    for i in range(max_touches):
        start = adj_spherical_init.shape[0] + (touch_chart_init_adj.shape[0] * i)
        end = adj_spherical_init.shape[0] + (touch_chart_init_adj.shape[0] * (i + 1))
        new_adj[start:end, start:end] = deepcopy(touch_chart_init_adj)
    adj = new_adj.astype(np.float32)

    # update adjacency matrix to allow communication between vision and touch charts
    for key in hash.keys():
        cur_verts = hash[key]
        if len(cur_verts) > 1:  # if there are multiple vertices, it means it is a boundary vertex
            for v1 in cur_verts:
                for v2 in cur_verts:  # vertices on the boundary of vision charts can communicate
                    adj[v1, v2] = 1
                    for c in central_points:  # touch and vision charts can communicate
                        adj[v1, c] = 1
                        adj[c, v1] = 1
    return adj

def adjacency_matrix(faces, verts, max_touches):
    """
    Compute the adjacency matrix of initial spherical mesh + touch charts and their intra-connections. Results are stored in torch.tensors.
    Params:
        - verts: initial spherical vertices + touch charts vertices
        - faces: initial spherical faces + touch charts faces
    Returns:
        - ad_info: dictionary containing torch tensors of keys 'original', 'adj'
            - 'original': normalised adjacency matrix for initial sphere, torch tensor shape(1824, 1824) 
            - 'adj': normalised adjacency matrix of initial sphere + touch charts, torch tensor shape().
    """
    adj_info = {}

    # get generic adjacency matrix for initial spherical chart
    adj_spherical_init = calc_adj(faces[:2304, :])   # 2304: number of faces in the initial chart
    adj_original = normalize_adj(deepcopy(adj_spherical_init))
    adj_info['original'] = torch.from_numpy(adj_original)
    # This combines the adjacency information of touch and vision charts
    # the output adj matrix has the first k rows corresponding to vision charts, and the last |V| - k
    # corresponding to touch charts. Similarly the first l faces are correspond to vision charts, and the
    # remaining correspond to touch charts
    adj = adj_fuse_touch(verts, adj_spherical_init, max_touches)
    adj = normalize_adj(adj)
    adj_info['adj'] = torch.from_numpy(adj)
    return adj_info

def translate_rotate_mesh(pos_wrld_list, rot_M_wrld_list, pointclouds_list, obj_initial_pos):
    """
    Given a pointcloud (workframe), the position of the TCP (worldframe), the rotation matrix (worldframe),
    it returns the pointcloud in worldframe. It assumes a default position of the object.

    Params:
        pos_wrld_list: (m, 3)
        rot_M_wrld_list: (m, 3, 3)
        pointclouds_list: pointcloud in workframe (m, number_points, 3)

    Returns:
    """
    a = rot_M_wrld_list @ pointclouds_list.transpose(0,2,1)
    b = a.transpose(0,2,1)
    c = pos_wrld_list[:, np.newaxis, :] + b
    pointcloud_wrld = c - obj_initial_pos
    return pointcloud_wrld

def draw_vertices_on_pb(vertices_wrld, color=[235, 52, 52]):
    color = np.array(color)/255
    color_From_array = np.full(shape=vertices_wrld.shape, fill_value=color)
    pb.addUserDebugPoints(
        pointPositions=vertices_wrld,
        pointColorsRGB=color_From_array,
        pointSize=1
    )

def get_mesh_z(obj_index):
    """
    Compute the mesh geometry and return the initial z-axis. This is to avoid that the object
    goes partially throught the ground.
    """
    filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
    verts, _ = mesh_from_urdf(filepath_obj)
    pointcloud_s = scale_pointcloud(np.array(verts))
    pointcloud_s_r = rotate_pointcloud(pointcloud_s)
    z_values = pointcloud_s_r[:, 2]
    height = (np.amax(z_values) - np.amin(z_values))
    return height/2

    # # worldframe coordinates. These do not take into account the initial obj position.
    # num_vertices, vertices_wrld = p.getMeshData(obj_id, 0)   
    # initial_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])   # WHY DO I NEED THIS ARBITRARY ORN INSTEAD OF OBJ ORN?
    # vertices_wrld = rotate_vector_by_quaternion(np.array(vertices_wrld), initial_orn) + initial_pos






