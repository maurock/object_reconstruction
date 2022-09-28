import numpy as np
import torch
from torch.utils.data import Dataset
import object_reconstruction.data.touch_charts as touch_charts
import os
from glob import glob
from object_reconstruction.utils.mesh_utils import translate_rotate_mesh

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TouchChartDataset(Dataset):
    def __init__(self, touch_charts_dir):
        """
        Check self.get_total_data() for details on touch_charts_dir and the required file tree.
        """
        touch_charts_dict = self.get_total_data(touch_charts_dir)
        self.data = touch_charts_dict
        for key in self.data.keys():
            self.data[key] = torch.tensor(self.data[key], dtype=torch.float32, device=device)
        return

    def __len__(self):
        return self.data['pointclouds'].shape[0]

    def __getitem__(self, idx):
        x = self.data['tactile_imgs'][idx, :, :]
        y = self.data['pointclouds'][idx, :, :]
        return x, y
    
    def get_total_data(self, touch_charts_dir):
        """
        Combine the touch_charts_gt for all the objects in `data/touch_charts/`.
        This method assumes the following tree:
        - data
        |   - touch_charts
        |   |   - {obj_index}, e.g. 101351
        |   |   |   - touch_charts_gt.npy
        Data is combined by vstacking all the arrays.

        Parameters:
            - touch_charts_dir = directory to the /touch_charts folder
        
        Returns:
            - full_data = np.array() containing all the touch_charts_gt.npy
        """
        print('Creating total dataset for touch chart prediction...')
        total_pointclouds = np.array([], dtype=np.float32).reshape(0, 2000, 3)
        total_tactile_imgs = np.array([], dtype=np.float32).reshape(0, 1, 256, 256)
        total_touch_charts_dict = dict()
        # Get paths of all touch_charts_gt.npy in touch_charts/
        filepaths = glob(os.path.join(touch_charts_dir, '*/touch_charts_gt.npy'))  
        for filepath in filepaths:
            touch_chart_dict = np.load(filepath, allow_pickle=True).item()
            total_pointclouds = np.vstack((total_pointclouds, touch_chart_dict['pointclouds']))
            total_tactile_imgs = np.vstack((total_tactile_imgs, touch_chart_dict['tactile_imgs']))
        total_touch_charts_dict['pointclouds'] = total_pointclouds
        total_touch_charts_dict['tactile_imgs'] = total_tactile_imgs
        print('Total dataset for touch chart prediction created.')

        return total_touch_charts_dict

    def _analyse_dataset(self):
        """
        Print information about full dataset
        """
        print(f"The dataset has shape {self.data['tactile_imgs'].shape}")
        print(f"The ground truth pointcloud for one instance has a shape {self.data['pointclouds'][0].shape}")
    

if __name__=='__main__':
    dataset = TouchChartDataset(os.path.dirname(touch_charts.__file__))
    print(dataset[0])
