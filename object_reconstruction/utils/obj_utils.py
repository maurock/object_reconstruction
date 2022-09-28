import trimesh
import os
import sys
from contextlib import contextmanager
import pybullet as pb
import numpy as np
from object_reconstruction.utils import mesh_utils

def load_obj(filename, p):
    # Load .OBJ in pybullet and using trimesh, si that we can load the vertices. PyBullet does not allow 
    # to use p.getMesh with .obj, but only with .urdf

    # example filename = os.path.join(os.path.dirname(__file__), '..', "stimuli/objects/627.obj")
    obj_mass = 0.1
    obj_position = [0.65, 0, 0.1]
    obj_orientation = [0, 0, 0, 1]
    scale_obj = 0.5
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                        fileName=filename,
                                        meshScale=[scale_obj, scale_obj, scale_obj])
    visual_id = p.createVisualShape(p.GEOM_MESH,
                                    fileName=filename,
                                    meshScale=[scale_obj, scale_obj, scale_obj])
    stimulus_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id,
                                  baseVisualShapeIndex=visual_id,
                                  basePosition=obj_position,
                                  baseOrientation=obj_orientation,
                                  baseMass=obj_mass)
    mesh = trimesh.load_mesh(file_obj=filename)

    return collision_shape_id, stimulus_id, mesh

"""Deal with b3Warning regarding missing links in the .URDF (SAPIEN)"""
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different