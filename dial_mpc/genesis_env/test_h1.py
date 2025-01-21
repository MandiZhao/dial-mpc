import torch
import numpy as np
import genesis as gs
import os 
import sys 

# xml_path = "/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/mjx_scene_h1_walk.xml"
# xml_path = "/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/mjx_h1_walk_real_feet.xml" # this only has feet collision enabled lmao
xml_path = "/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/h1_real_feet.xml" 

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer    = False,
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3.5, -1.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        res           = (1920, 1080),
    ),
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
        constraint_solver = gs.constraint_solver.Newton, # Newton solver is faster than the default conjugate gradient (CG) solver.
    ),
) 
plane = scene.add_entity(gs.morphs.Plane())
entity = scene.add_entity(
    gs.morphs.MJCF(file=xml_path),
)
B = 1
scene.build(n_envs=B, env_spacing=(2.0, 2.0))

for i in range(10):
    scene.step()

# check the parsed joint range and actuator ctrl range!
force_range = entity.get_dofs_force_range()

# has parsing bugs, manually set:
""" 
<actuator>
    <motor class="h1" name="left_hip_yaw" joint="left_hip_yaw" ctrlrange="-200 200"/>
    <motor class="h1" name="left_hip_roll" joint="left_hip_roll" ctrlrange="-200 200"/>
    <motor class="h1" name="left_hip_pitch" joint="left_hip_pitch" ctrlrange="-200 200"/>
    <motor class="h1" name="left_knee" joint="left_knee" ctrlrange="-300 300"/>
    <motor class="h1" name="left_ankle" joint="left_ankle" ctrlrange="-40 40"/>
    <motor class="h1" name="right_hip_yaw" joint="right_hip_yaw" ctrlrange="-200 200"/>
    <motor class="h1" name="right_hip_roll" joint="right_hip_roll" ctrlrange="-200 200"/>
    <motor class="h1" name="right_hip_pitch" joint="right_hip_pitch" ctrlrange="-200 200"/>
    <motor class="h1" name="right_knee" joint="right_knee" ctrlrange="-300 300"/>
    <motor class="h1" name="right_ankle" joint="right_ankle" ctrlrange="-40 40"/>
    <motor class="h1" name="torso" joint="torso" ctrlrange="-200 200"/>
    <motor class="h1" name="left_shoulder_pitch" joint="left_shoulder_pitch" ctrlrange="-40 40"/>
    <motor class="h1" name="left_shoulder_roll" joint="left_shoulder_roll" ctrlrange="-40 40"/>
    <motor class="h1" name="left_shoulder_yaw" joint="left_shoulder_yaw" ctrlrange="-18 18"/>
    <motor class="h1" name="left_elbow" joint="left_elbow" ctrlrange="-18 18"/>
    <motor class="h1" name="right_shoulder_pitch" joint="right_shoulder_pitch" ctrlrange="-40 40"/>
    <motor class="h1" name="right_shoulder_roll" joint="right_shoulder_roll" ctrlrange="-40 40"/>
    <motor class="h1" name="right_shoulder_yaw" joint="right_shoulder_yaw" ctrlrange="-18 18"/>
    <motor class="h1" name="right_elbow" joint="right_elbow" ctrlrange="-18 18"/>
</actuator>
"""
# parse the above string
range_dict = {
    "left_hip_yaw": [-200, 200],
    "left_hip_roll": [-200, 200],
    "left_hip_pitch": [-200, 200],
    "left_knee": [-300, 300],
    "left_ankle": [-40, 40],
    "right_hip_yaw": [-200, 200],
    "right_hip_roll": [-200, 200],
    "right_hip_pitch": [-200, 200],
    "right_knee": [-300, 300],
    "right_ankle": [-40, 40],
    "torso": [-200, 200],
    "left_shoulder_pitch": [-40, 40],
    "left_shoulder_roll": [-40, 40],
    "left_shoulder_yaw": [-18, 18],
    "left_elbow": [-18, 18],
    "right_shoulder_pitch": [-40, 40],
    "right_shoulder_roll": [-40, 40],
    "right_shoulder_yaw": [-18, 18],
    "right_elbow": [-18, 18],
}

actuated_joints = [joint for joint in entity.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
actuated_dof_names = [joint.name for joint in actuated_joints]
actuated_dof_idxs = [joint.dof_idx_local for joint in actuated_joints]
custom_force_range = np.zeros((len(actuated_dof_idxs), 2))
for i, name in enumerate(actuated_dof_names):
    custom_force_range[i] = range_dict[name]

lower = torch.tensor(custom_force_range[:, 0], dtype=torch.float32)
upper = torch.tensor(custom_force_range[:, 1], dtype=torch.float32)
entity.set_dofs_force_range(lower, upper, dofs_idx_local=actuated_dof_idxs)
print("Custom force range set!", entity.get_dofs_force_range())
limits = entity.get_dofs_limit()
breakpoint()
