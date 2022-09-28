import math
import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from PIL import Image
import os
from object_reconstruction.utils.mesh_utils import *
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# basic CNN layer template
def CNN_layer(f_in, f_out, k, stride=1, simple=False, padding=1):
    layers = []
    if not simple:
        layers.append(nn.BatchNorm2d(int(f_in)))
        layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(int(f_in), int(f_out), kernel_size=k, padding=padding, stride=stride)
    )
    return nn.Sequential(*layers)


# Class for defroming the charts into the traget shape
class Deformation(nn.Module):
    def __init__(
        self, touch_chart_dir, args):
        super(Deformation, self).__init__()
        # load adjacency matrix
        self.touch_chart_dir = touch_chart_dir
        self.adj_info_path = os.path.join(self.touch_chart_dir, 'adj_info.npy')
        self.adj_info = np.load(self.adj_info_path, allow_pickle=True).item()
        self.adj_info['original'] = self.adj_info['original'].to(device)
        self.adj_info['adj'] = self.adj_info['adj'].to(device)

        # load initial sphere mesh and calculate the number of vertices
        init_sphere_path = os.path.join(self.touch_chart_dir, 'vision_charts.obj')
        init_sphere_verts, _ = load_mesh_touch(init_sphere_path) # this returns torch.tensors
        init_sphere_verts = init_sphere_verts * args.initial_sphere_dimension  # reduce the initial size of the sphere 
        self.init_sphere_length = init_sphere_verts.shape[0]

        #self.initial_positions = initial_positions
        self.args = args
        
        # if no image features fix the feature size at 50
        input_size = self.args.input_size

        # add positional and mask enocoder and GCN deformation networks
        self.positional_encoder = Positional_Encoder(input_size, self.args.num_layers_embedding)
        self.mask_encoder = Mask_Encoder(input_size)
        self.mesh_deform_1 = GCN(
            input_size, args).to(device)
        self.mesh_deform_2 = GCN(input_size, args).to(device)

    def forward(self, batch):
        """
        batch contains (verts, faces, mask, obj_pointcloud)
        """
        ##### first iteration #####
        # if we are using only touch then we need to use touch information immediately
        # use touch information
        vertices = batch[0].clone()
        mask = batch[2].clone()
        positional_features = self.positional_encoder(vertices)
        mask_features = self.mask_encoder(mask)
        vertex_features = positional_features + mask_features

        # perfrom the first deformation
        update = self.mesh_deform_1(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :self.init_sphere_length] = vertices[:, :self.init_sphere_length] + update[:, :self.init_sphere_length]

        ##### second loop #####
        # add touch information if not already present
        positional_features = self.positional_encoder(vertices)
        vertex_features = positional_features + mask_features

        # perfrom the second deformation
        update = self.mesh_deform_2(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :self.init_sphere_length] = vertices[:, :self.init_sphere_length] + update[:, :self.init_sphere_length]

        ##### third loop #####
        positional_features = self.positional_encoder(vertices)
        mask_features = self.mask_encoder(mask)
        vertex_features = positional_features + mask_features

        # perfrom the third deformation
        update = self.mesh_deform_2(vertex_features, self.adj_info)
        # update positions of vision charts only
        vertices[:, :self.init_sphere_length] = vertices[:, :self.init_sphere_length] + update[:, :self.init_sphere_length]

        return vertices, mask


# Graph convolutional network class for predicting mesh deformation
class GCN(nn.Module):
    def __init__(self, input_features, args, ignore_touch_matrix=False):
        super(GCN, self).__init__()
        #
        self.ignore_touch_matrix = ignore_touch_matrix
        self.num_layers = args.num_GCN_layers
        # define output sizes for each GCN layer
        hidden_values = (
            [input_features]
            + [args.hidden_GCN_size for _ in range(self.num_layers - 1)]
            + [3]
        )

        # define layers
        layers = []
        for i in range(self.num_layers):
            layers.append(
                GCN_layer(
                    hidden_values[i],
                    hidden_values[i + 1],
                    args.cut,
                    do_cut=i < self.num_layers - 1,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, features, adj_info):
        if self.ignore_touch_matrix:
            adj = adj_info["original"]
        else:
            adj = adj_info["adj"]

        # iterate through GCN layers
        for i in range(self.num_layers):
            activation = F.relu if i < self.num_layers - 1 else lambda x: x
            features = self.layers[i](features, adj, activation)
            if torch.isnan(features).any():
                print(features)
                print("here", i, self.num_layers)
                input()

        return features


# Graph convolutional network layer
class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, cut=0.33, do_cut=True):
        super(GCN_layer, self).__init__()
        self.weight = Parameter(torch.Tensor(1, in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.cut_size = cut
        self.do_cut = do_cut

    def reset_parameters(self):
        stdv = 6.0 / math.sqrt((self.weight.size(1) + self.weight.size(0)))
        stdv *= 0.3
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-0.05, 0.05)

    def forward(self, features, adj, activation):
        features = torch.matmul(features, self.weight)
        # uf we want to only share a subset of features with neighbors
        if self.do_cut:
            length = round(features.shape[-1] * self.cut_size)
            output = torch.matmul(adj, features[:, :, :length])
            output = torch.cat((output, features[:, :, length:]), dim=-1)
            output[:, :, :length] += self.bias[:length]
        else:
            output = torch.matmul(adj, features)
            output = output + self.bias

        return activation(output)


# encode the positional information of vertices using Nerf Embeddings
class Positional_Encoder(nn.Module):
    def __init__(self, input_size, num_layers_embedding):
        super(Positional_Encoder, self).__init__()
        self.num_layers_embedding = num_layers_embedding
        layers = []
        layers.append(
            nn.Linear(self.num_layers_embedding * 2 * 3 + 3, input_size // 4)
        )  # 10 nerf layers + original positions
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 4, input_size // 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(input_size // 2, input_size))
        self.model = nn.Sequential(*layers)

    # apply nerf embedding of the positional information
    def nerf_embedding(self, points):
        embeddings = []
        for i in range(self.num_layers_embedding):
            if i == 0:
                embeddings.append(torch.sin(np.pi * points))
                embeddings.append(torch.cos(np.pi * points))
            else:
                embeddings.append(torch.sin(np.pi * 2 * i * points))
                embeddings.append(torch.cos(np.pi * 2 * i * points))
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings

    def forward(self, positions):
        shape = positions.shape
        positions = positions.contiguous().view(shape[0] * shape[1], -1)
        # combine nerf embedding with origional positions
        positions = torch.cat((self.nerf_embedding((positions)), positions), dim=-1)
        embedding = self.model(positions).view(shape[0], shape[1], -1)
        return embedding


# make embedding token of the mask information for each vertex
class Mask_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Mask_Encoder, self).__init__()
        layers_mask = []
        layers_mask.append(nn.Embedding(4, input_size))
        self.model = nn.Sequential(*layers_mask)

    def forward(self, mask):
        shape = mask.shape
        mask = mask.contiguous().view(-1, 1)
        embeding_mask = self.model(mask.long()).view(shape[0], shape[1], -1)
        return embeding_mask