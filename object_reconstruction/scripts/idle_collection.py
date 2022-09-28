import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil
import os
from tactile_gym_sim2real.data_collection.sim.cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path
from object_reconstruction.utils.samples_utils import *
from object_reconstruction.utils.mesh_utils import *
from object_reconstruction.utils.obj_utils import *
import trimesh 
from object_reconstruction.touch.train import *
from glob import glob
import object_reconstruction.data.objects as objects
import object_reconstruction.data.obj_pointcloud as obj_pointcloud

"""
In this script, a robot arm is in idle state. Used to take screenshot for the report.
"""
def main(args):
    time_step = 1. / 960  # low for small objects

    if args.show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0, 0, -10)
    pb.setPhysicsEngineParameter(fixedTimeStep=time_step,
                                 numSolverIterations=150,  # 150 is good but slow
                                 numSubSteps=1,
                                 contactBreakingThreshold=0.0005,
                                 erp=0.05,
                                 contactERP=0.05,
                                 frictionERP=0.2,
                                 # need to enable friction anchors (maybe something to experiment with)
                                 solverResidualThreshold=1e-7,
                                 contactSlop=0.001,
                                 globalCFM=0.0001)

    if args.show_gui:
        # set debug camera position
        cam_dist = 0.5
        cam_yaw = 90
        cam_pitch = -25
        cam_pos = [0.65, 0, 0.025]
        pb.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

    # load the environment
    plane_id = pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    )

    # set up tactip
    tactip_type = 'standard'
    tactip_core = 'no_core'
    tactip_dynamics = {}

    # setup workspace
    workframe_pos = [0.65, 0.0, 0.35]  # relative to world frame
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]  # relative to world frame
    
    robot = CRIRobotArm(
        pb,
        workframe_pos=workframe_pos,
        workframe_rpy=workframe_rpy,
        image_size=[256, 256],
        arm_type="ur5",
        tactip_type=tactip_type,
        tactip_core=tactip_core,
        tactip_dynamics=tactip_dynamics,
        show_gui=args.show_gui,
        show_tactile=args.show_tactile
    )
    print(f'Robot ID: {robot.robot_id}')

    time.sleep(2)
    
    list_objects = [filepath.split('/')[-1] for filepath in glob(os.path.join(os.path.dirname(objects.__file__), '*'))]
    list_objects.remove('__init__.py')
    list_objects.remove('__pycache__')

    for idx, obj_index in enumerate(list_objects):
        if idx==1:
            break
        print(f"Collecting data... Object index: {obj_index}     {idx+1}/{len(list_objects)} ")
        stimulus_pos = [0.65, 0.0, 0.025]
        stimulus_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        scale_obj = 0.1
        with suppress_stdout():          # to suppress b3Warning
            stimulus_id = pb.loadURDF(
                #add_assets_path("rl_env_assets/exploration/edge_follow/edge_stimuli/square/square.urdf"),
                os.path.join(os.path.dirname(__file__), '..', f"data/objects/{obj_index}/mobility.urdf"),
                stimulus_pos,
                stimulus_orn,
                useFixedBase=True,
                flags=pb.URDF_INITIALIZE_SAT_FEATURES,
                globalScaling=scale_obj
            )
            print(f'PyBullet object ID: {stimulus_id}')


        # if OBJ instead of URDF
        #collision_shape_id, stimulus_id, mesh = load_obj(filename, p)
        
        initial_pos_wrk = robot.arm.worldframe_to_workframe([0.65, 0.0, 1.2], [0, 0, 0])[0]

        pos, _ = robot.arm.worldframe_to_workframe([0.65, 0, 0.2], [0, 0, 0])
        robot.move_linear(pos , [0, 0, 0])

        idx_x = pb.addUserDebugParameter("x",-2,2,0.65)
        idx_y = pb.addUserDebugParameter("y",-2,2,0)
        idx_z = pb.addUserDebugParameter("z",0,1,0.2)
        idx_rot_x_wrk = pb.addUserDebugParameter("rot_x_wrk",-3.2,3.2,0)
        idx_rot_y_wrk = pb.addUserDebugParameter("rot_y_wrk",-3.2,3.2,0)
        idx_rot_z_wrk = pb.addUserDebugParameter("rot_z_wrk",-3.2,3.2,0)
        idx_rot_x_wrld = pb.addUserDebugParameter("rot_x_wrld",-3.2,3.2,0)
        idx_rot_y_wrld = pb.addUserDebugParameter("rot_y_wrld",-3.2,3.2,0)
        idx_rot_z_wrld = pb.addUserDebugParameter("rot_z_wrld",-3.2,3.2,0)
        print(idx_x, idx_y, idx_z, idx_rot_x_wrk, idx_rot_y_wrk, idx_rot_z_wrk, idx_rot_x_wrld, idx_rot_y_wrld, idx_rot_z_wrld)
        pb.setRealTimeSimulation(1)
        while(True):
            #robot._pb.stepSimulation()  
            #time.sleep(1./240)
            try:
                x = pb.readUserDebugParameter(idx_x)
                y = pb.readUserDebugParameter(idx_y)
                z = pb.readUserDebugParameter(idx_z)
                rot_x_wrk = pb.readUserDebugParameter(idx_rot_x_wrk)
                rot_y_wrk = pb.readUserDebugParameter(idx_rot_y_wrk)
                rot_z_wrk = pb.readUserDebugParameter(idx_rot_z_wrk)
                rot_x_wrld = pb.readUserDebugParameter(idx_rot_x_wrld)
                rot_y_wrld = pb.readUserDebugParameter(idx_rot_y_wrld)
                rot_z_wrld = pb.readUserDebugParameter(idx_rot_z_wrld)
            except:
                print('Error', idx_x, idx_y, idx_z, idx_rot_x_wrk, idx_rot_y_wrk, idx_rot_z_wrk)
            
            pos, orn_wrk = robot.arm.worldframe_to_workframe([x, y, z], [rot_x_wrld, rot_y_wrld, rot_z_wrld])
            robot.move_linear(pos , [rot_x_wrk, rot_y_wrk, rot_z_wrk])

            # Draw lines for object normals
            pos_tcp_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[0]
            orn_tcp_wrld = robot.arm.get_current_TCP_pos_vel_worldframe()[2]
            pos_tcp_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[0]
            orn_tcp_wrk = robot.arm.get_current_TCP_pos_vel_workframe()[2]      
            rot_wrld = np.array(pb.getMatrixFromQuaternion(orn_tcp_wrld)).reshape(3,3)
            x_axis_wrld = (rot_wrld @ np.array([1, 0, 0]).reshape(3,1)).transpose(1,0)[0] + pos_tcp_wrld
            y_axis_wrld = (rot_wrld @ np.array([0, 1, 0]).reshape(3,1)).transpose(1,0)[0] + pos_tcp_wrld
            z_axis_wrld = (rot_wrld @ np.array([0, 0, 1]).reshape(3,1)).transpose(1,0)[0] + pos_tcp_wrld
            print(f'x_axis: {x_axis_wrld} \ny_axis: {y_axis_wrld} \nz_axis: {z_axis_wrld}')
            pb.addUserDebugLine(pos_tcp_wrld, x_axis_wrld, lifeTime=0.5)
            pb.addUserDebugLine(pos_tcp_wrld, y_axis_wrld, lifeTime=0.5)
            pb.addUserDebugLine(pos_tcp_wrld, z_axis_wrld, lifeTime=0.5)


    # for i in range (10000):
    #     robot._pb.stepSimulation()  
    #     time.sleep(1./240.)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_gui", type=bool, default=True, help="Show PyBullet GUI"
    )
    parser.add_argument(
        "--show_tactile", type=bool, default=False, help="Show tactile image"
    )
    parser.add_argument(
        "--debug_show_full_mesh", type=bool, default=False, help="Show mesh obtained from first raycasting"
    )
    parser.add_argument(
        "--debug_show_mesh_wrk", type=bool, default=False, help="Show mesh obtained from applying the pivot ball technique on 25 vertices wrt workframe"
    )
    parser.add_argument(
        "--debug_show_mesh_wrld", type=bool, default=False, help="Show mesh obtained from applying the pivot ball technique on 25 vertices wrt worldframe"
    )
    parser.add_argument(
        "--debug_contact_points", type=bool, default=False, help="Show contact points on Plotly"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of samplings on the objects"
    )
    parser.add_argument(
        "--debug_rotation", type=int, default=50, help="Number of samplings on the objects"
    )
    args = parser.parse_args()

    args.num_samples = 5

    main(args)