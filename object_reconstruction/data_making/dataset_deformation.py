import numpy as np
import torch
from torch.utils.data import Dataset
import object_reconstruction.data.touch_charts as touch_charts
import object_reconstruction.data.obj_pointcloud as obj_pointcloud
import os
from glob import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeformationDataset(Dataset):
    """
    Check self.get_total_data() for details on touch_vision charts and the required file tree.
    """
    def __init__(self, path):
        vision_touch_dict = self.get_total_data(path)
        self.data = vision_touch_dict
        for key in self.data.keys():
            self.data[key] = torch.from_numpy(self.data[key]).to(device)
        return

    def __len__(self):
        return self.data['verts'].shape[0]

    def __getitem__(self, idx):
        verts = self.data['verts'][idx, :, :]
        faces = self.data['faces'][idx, :, :]
        mask =  self.data['mask'][idx, :]
        obj_pointcloud = self.data['obj_pointcloud'][idx, :, :]
        return verts, faces, mask, obj_pointcloud
    
    def get_total_data(self, touch_charts_dir, max_touches=5):
        """
        Combine the touch_vision charts for all the objects in `data/touch_charts/`.
        This method assumes the following tree:
        - data
        |   - touch_charts
        |   |   - {obj_index}, e.g. 101351
        |   |   |   - touch_vision.npy

        Parameters:
            - touch_charts_dir = directory to the /touch_charts folder
        
        Returns:
            - full_data = np.array() containing all the touch_charts_gt.npy
        """
        print('Creating total dataset for deformation prediction...')
        # Set dimensions of arrays as function of max_touches
        # num faces for initial sphere (2304) + args.max_touches * faces in each touch chart (32)
        num_faces = 2304 + max_touches * 32
        # num verts for initial sphere (1824) + args.max_touches * verts in each touch chart (25)
        num_verts = 1824 + max_touches * 25    
        # initialize np arrays for colelcting data
        total_verts = np.array([], dtype=np.float32).reshape(0, num_verts, 3)     
        total_faces = np.array([], dtype=np.int64).reshape(0, num_faces, 3) 
        total_mask = np.array([], dtype=np.float32).reshape(0, num_verts)
        total_obj_pointcloud = np.array([], dtype=np.float32).reshape(0, 2000, 3)
        total_dict = dict()
        # Get paths of all touch_charts_gt.npy in touch_charts/
        filepaths = glob(os.path.join(touch_charts_dir, '*/touch_vision.npy'))  
        for filepath in filepaths:
            touch_chart_dict = np.load(filepath, allow_pickle=True).item()
            data_size = touch_chart_dict['verts'].shape[0]   # num elements for this object
            total_verts = np.vstack((total_verts, touch_chart_dict['verts']))
            total_faces = np.vstack((total_faces, touch_chart_dict['faces']))
            total_mask = np.vstack((total_mask, touch_chart_dict['mask']))
            # object full pointcloud is not stored in the vision_touch_charts, but it can be retrieved from obj_pointcloud/obj_pointcloud.npy
            obj_index = filepath.split('/')[-2]
            obj_pointcloud_path = os.path.join(os.path.dirname(obj_pointcloud.__file__), obj_index, 'obj_pointcloud.npy')
            obj_pointcloud_data = np.load(obj_pointcloud_path, allow_pickle=True)[np.newaxis, :, :]
            total_obj_pointcloud = np.vstack((total_obj_pointcloud, np.tile(obj_pointcloud_data.astype(np.float32), (data_size, 1, 1))))  # duplicate obj_pointcloud for all the elements in this object

        # Store total informaiton to dictionary
        total_dict['verts'] = total_verts
        assert total_verts[0].dtype == np.float32, f'verts dtype required is np.float32, found {total_verts[0].dtype} instead'
        total_dict['faces'] = total_faces
        assert total_faces[0].dtype == np.int64, f'faces dtype required is np.int64, found {total_faces[0].dtype} instead'
        total_dict['mask'] = total_mask
        assert total_mask[0].dtype == np.float32, f'mask dtype required is np.float32, found {total_mask[0].dtype} instead'
        total_dict['obj_pointcloud'] = total_obj_pointcloud
        assert total_obj_pointcloud[0].dtype == np.float32, f'obj pointcloud dtype required is np.float32, found {total_obj_pointcloud[0].dtype} instead'
       
        print('Total dataset for deformation prediction created.')

        return total_dict
    

if __name__=='__main__':
    dataset = DeformationDataset(os.path.dirname(touch_charts.__file__))
    print(dataset[0])