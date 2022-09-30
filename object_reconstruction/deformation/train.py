import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

import torch.optim as optim
from pterotactyl.reconstruction.vision import model
from pterotactyl.utility import utils
from pterotactyl.utility import data_loaders
import pterotactyl.objects as objects
from pterotactyl import pretrained

import object_reconstruction.data.touch_charts as touch_charts
import object_reconstruction.data.checkpoints as checkpoints
from object_reconstruction.utils import misc_utils
from object_reconstruction.data_making.dataset_deformation import DeformationDataset 
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import model
import plotly.graph_objects as go
from datetime import datetime
import json
from datetime import datetime
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():

    def __init__(self, args):
        # set random seeds
        misc_utils.set_seeds(42)

        # set paths
        self.touch_chart_dir = os.path.dirname(touch_charts.__file__)
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M')   # timestamp to use for logging data
        self.log_train_dir = os.path.join(os.path.dirname(checkpoints.__file__), 'deformation_model', self.timestamp_run)
        if not os.path.exists(self.log_train_dir):
            os.mkdir(self.log_train_dir)

        self.pretrain_path = os.path.join(os.path.dirname(checkpoints.__file__), 'deformation_model', '30_08_2354', 'weights.pt')
        
        # set initial values
        self.epoch = 0
        self.args = args
        self.results = dict()
    
    def __call__(self):
        if self.args.log_info_train:         # log info train
            # Create log. This will be populated with settings, losses, etc..
            self.log_path = os.path.join(self.log_train_dir, "settings.txt")
            args_dict = vars(self.args)  # convert args to dict to write them as json
            with open(self.log_path, mode='a') as log:
                log.write('Settings:\n')
                log.write(json.dumps(args_dict).replace(', ', ',\n'))
                log.write('\n\n')

        if self.args.visualise_deformation_train:     # store file to visualise deformation
            self.debug_visualisation = np.array([]).reshape(0, 1949, 3)
            self.debug_visualisation_path = os.path.join(self.log_train_dir, f"debug_visualisation_{self.timestamp_run}.npy")

        self.model = model.Deformation(self.touch_chart_dir, self.args).to(device)
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.args.lr_multiplier)

        if self.args.pretrained:
            self.model.load_state_dict(torch.load(self.pretrain_path))

         # get data
        train_loader, valid_loaders = self.get_loaders()

        # Variables to store results
        self.results['train'] = []
        self.results['val'] = []

        # train and validate
        best_val = 1000000
        patience = 0
        for epoch in range(0, self.args.epochs):
            # Learning rate scheduling
            if patience == 20:   # every 20 epochs, if val does not improve, reduce learning rate
                patience = 0
                self.scheduler.step()
            self.epoch = epoch
            self.train(train_loader)
            with torch.no_grad():
                val_loss = self.validate(valid_loaders)
                if val_loss < best_val:
                    patience = 0
                    best_val = deepcopy(val_loss)       
                patience+=1       
            # self.check_values()

    def get_loaders(self):
        full_dataset = DeformationDataset(self.touch_chart_dir)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_data, val_data = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True
            )
        val_loader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True
            )
        return train_loader, val_loader

    def train(self, data):
        total_loss = 0
        iterations = 0

        #params = list(self.model.parameters())
        #self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)
        self.model.train()

        for k, batch in enumerate(tqdm(data, smoothing=0)):
            self.optimizer.zero_grad()
            # initialize data
            gt_obj_pointcloud = batch[3].to(device)
            verts = self.model(batch)[0]
            loss = utils.chamfer_distance(
                verts[:1824], batch[1][0][:2304], gt_obj_pointcloud, num=self.args.number_points
            )
            loss = self.args.loss_coeff * loss.mean()
            # backprop
            loss.backward()
            self.optimizer.step()
            # log
            total_loss += loss.item()
            iterations += 1.0
        print(
            "train_loss", {total_loss / iterations}, self.epoch
        )
        self.results['train'].append(total_loss/iterations)
        np.save(os.path.join(self.log_train_dir, 'results_dict.npy'), self.results)
        torch.save(self.model.state_dict(), os.path.join(self.log_train_dir, 'weights.pt'))

        if self.args.log_info_train:
            with open(self.log_path, mode='a') as log:
                log.write(f'Epoch {self.epoch}, Train loss: {total_loss / iterations} \n')
    
    def validate(self, data):
        iterations = 0
        total_loss = 0
        self.model.eval()
        for k, batch in enumerate(tqdm(data, smoothing=0)):
            # initialize data
            gt_obj_pointcloud = batch[3].to(device)
            verts = self.model(batch)[0]

            loss = utils.chamfer_distance(
                verts, batch[1][0], gt_obj_pointcloud, num=self.args.number_points
            )
            loss = self.args.loss_coeff * self.args.lr_multiplier * loss.mean()

            # log
            total_loss += loss.item()
            iterations += 1.0

            if (self.args.visualise_deformation_train) and (iterations==1.0):   # debug: show deformation FIRST object in FIRST batch
                # vert: torch.tensor(), shape(1949, 3)
                vert_np = verts[0].cpu().numpy()[np.newaxis, :, :]
                self.debug_visualisation = np.vstack((self.debug_visualisation, vert_np))
                np.save(self.debug_visualisation_path, self.debug_visualisation)

        print(
            "val_loss", {total_loss / iterations}, self.epoch
        )
        self.results['val'].append(total_loss/iterations)
        if self.args.log_info_train:
            with open(self.log_path, mode='a') as log:
                log.write(f'Epoch {self.epoch}, Val loss: {total_loss / iterations} \n')
        
        return total_loss / iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--num_GCN_layers",
        type=int,
        default=20,
        help="Number of GCN layers in the mesh deformation network.",
    )
    parser.add_argument(
        "--hidden_GCN_size",
        type=int,
        default=300,
        help="Size of the feature vector for each GCN layer in the mesh deformation network.",
    )
    parser.add_argument(
        "--cut",
        type=float,
        default=0.33,
        help="The shared size of features in the GCN.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.00015, help="Initial learning rate."
    )
    parser.add_argument(
        "--number_points",
        type=int,
        default=30000,
        help="number of points sampled for the chamfer distance.",
    )
    parser.add_argument(
        "--loss_coeff", type=float, default=1000.0, help="Coefficient for loss term."
    )
    parser.add_argument(
        '--visualise_deformation_train', type=bool, default=False, help="Plot the deformed spherical mesh over training epochs."
    )
    parser.add_argument(
        '--log_info_train', type=bool, default=True, help="Store info about training."
    )
    parser.add_argument(
        '--input_size', type=int, default=50, help="Store info about training."
    )
    parser.add_argument(
        '--num_layers_embedding', type=int, default=10, help="Store info about training."
    )
    parser.add_argument(
        '--pretrained', type=bool, default=False, help="Store info about training."
    )  
    parser.add_argument(
        '--initial_sphere_dimension', type=float, default=0.5, help="Multiplier for the initial sphere dimension. Number smaller than 1 result in a smaller initial sphere."
    )  
    parser.add_argument(
        '--lr_multiplier', type=float, default=1.0, help="Multiplier for the learning rate scheduling"
    )  
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer()
