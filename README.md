# Shape Reconstruction using Optical Tactile Sensing
This repository provides an adaptation of the approach proposed in [Active 3D Shape Reconstruction using Vision and Touch](https://github.com/facebookresearch/Active-3D-Vision-and-Touch). The original paper describes a method to reconstruct 3D shapes from the interaction between a robot and a set of 3D shapes. In this repository, we adapted this technique to the vision-based tactile sensor [TacTip](https://www.liebertpub.com/doi/10.1089/soro.2017.0052).

We used the open-source robot learning library [Tactile-Gym](https://github.com/ac-93/tactile_gym) to control the robot and collect data for shape reconstruction. Tactile-Gym is based on the physical simulator PyBullet, and provides a suite of learning environments for highly-efficient tactile image rendering.

If you find this code useful, please consider citing the following in your BibTex entry:
```
@misc{comi2022active,
author = {Mauro Comi},
title = {{3D Shape Reconstruction using Optical Tactile Sensing}},
howpublished = {\url{https://github.com/maurock/object_reconstruction}},
year = {2022}
}

@article{smith2021active,
  title={Active 3D Shape Reconstruction from Vision and Touch},
  author={Smith, Edward J and Meger, David and Pineda, Luis and Calandra, Roberto and Malik, Jitendra and Romero, Adriana and Drozdzal, Michal},
  journal={arXiv preprint arXiv:2107.09584},
  year={2021}
}
```

If you are using the Tactile-Gym library, please refer to [the source website](https://github.com/ac-93/tactile_gym) for citation. 
# Content
- [Installation](#installation)

# Installation
This repository was developed using Pyton=3.8. It was tested on macOS Monterey M1 (tested on CPU, but it can run on a GPU), Ubuntu 20.04 (GPU), and CentOS 7 (GPU). Some python libraries vary slightly depending on the OS you are using. Later in this README, I provide OS-dependent installation instructions for those libraries. 

To clone this repository, install it, and create a new environment:
```
conda create -n optical_reconstruction python=3.8
conda activate optical_reconstruction
git clone https://github.com/maurock/object_reconstruction.git
cd object_reconstruction
pip install -e .
```
To create the directories needed to collect data and run tests, run
```
bash create_directories.sh
```
In addition, you need to install the `Tactile-Gym` library (branch `active_reconstruction`):
```
pip install -e "git+https://github.com/ac-93/tactile_gym.git@active_reconstruction#egg=tactile_gym"
cd src/tactile-gym
python setup.py install
```
Additional libraries required:
```
conda install plotly scikit-learn==1.0.2 -c conda-forge
pip install open3d==0.15.1 pyrsistent==0.18.1 trimesh==3.10.2
```
### On macOS (CPU)
On macOS, this repository uses Python 3.8, PyTorch 1.9 (CPU, but GPU should also work), PyTorch3D 0.7 (CPU-only). 
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
### On Ubuntu (GPU)
Pytorch3D supports the GPU on Ubuntu
```
FORCE_CUDA=1 conda install pytorch3d -c pytorch3d -c anaconda -c pytorchconda install -c fvcore -c iopath -c conda-forge fvcore iopath
```





