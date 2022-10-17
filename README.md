# Shape Reconstruction using Optical Tactile Sensing
This repository provides an adaptation of the approach proposed in [Active 3D Shape Reconstruction using Vision and Touch](https://github.com/facebookresearch/Active-3D-Vision-and-Touch). The original paper described a method to reconstruct 3D shapes from the interaction between a robot and a set of 3D shapes. In their approach, the authors collected data using the vision-based tactile sensor DIGIT. In this repository, we adapted this technique to the vision-based tactile sensor [TacTip](https://www.liebertpub.com/doi/10.1089/soro.2017.0052).

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

If you are using the Tactile-Gym library, please refer to [its source](https://github.com/ac-93/tactile_gym) for citation. 

# Installation
This code was tested on macOS Monterey M1 (tested on CPU, but it can run on a GPU), Ubuntu 20.04 (GPU), and CentOS 7 (GPU). The version of the pyhon libraries used in this project vary slightly depending on the OS. On macOS, it uses Python 3.8, PyTorch 1.9, PyTorch3D 0.6.1. To install these required libraries on Ubuntu:
```
conda create -n optical_reconstruction python=3.8
conda activate optical_reconstruction
conda install -c pytorch pytorch=1.9.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
In addition, the `Tactile-Gym` library (branch `active_reconstruction`) needs to be installed:
```
pip install -e "git+https://github.com/ac-93/tactile_gym.git@active_reconstruction#egg=tactile_gym"
```
### On macOS
On macOS, it uses Python 3.8, PyTorch 1.9, PyTorch3D 0.6.1
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```




