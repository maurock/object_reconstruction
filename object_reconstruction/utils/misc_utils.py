from random import random
import numpy as np
import torch
import random
from PIL import Image
import pybullet as pb

# set seeds for consistency
def set_seeds(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	random.seed(seed)

def render_scene():
	# Render image
	projectionMatrix = pb.computeProjectionMatrixFOV(
		fov=55.0,
		aspect=1.0,
		nearVal=0.1,
		farVal=3.1)
	viewMatrix = pb.computeViewMatrix(
		cameraEyePosition=[0.95, 0, 0.2],
		cameraTargetPosition=[0.65, 0, 0],
		cameraUpVector=[0, 0, 1])
	width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
		width=224, 
		height=224,
		viewMatrix=viewMatrix,
		projectionMatrix=projectionMatrix)
	
	img = Image.fromarray(rgbImg, 'RGBA')
	img.show()