from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, Union, List
import torch 
import numpy as np 
import genesis as gs
from genesis.utils.geom import quat_to_xyz, inv_transform_by_quat, transform_by_quat 
from dataclasses import dataclass
 
# @torch.jit.script
def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        # Compute the angle
        angle = (t + torch.pi - footphase) % (2 * torch.pi) - torch.pi
        
        # Scale angle if duty_ratio < 1
        angle = torch.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        
        # Clip the angle to [-pi/2, pi/2]
        clipped_angle = torch.clamp(angle, -torch.pi / 2, torch.pi / 2)
        
        # Compute the cosine value, set to 0 if duty_ratio >= 1
        value = torch.where(duty_ratio < 1, torch.cos(clipped_angle), torch.zeros_like(clipped_angle))
        
        # Set small values to 0
        final_value = torch.where(torch.abs(value) >= 1e-6, torch.abs(value), torch.zeros_like(value))
        return final_value

    # Compute the time-dependent phase input for step_height
    time_phases = time * 2 * torch.pi * cadence + torch.pi
    foot_phases = 2 * torch.pi * phases
    
    # Vectorized computation using torch.vmap equivalent (batch processing with broadcasting)
    h_steps = amplitude * torch.stack([
        step_height(time_phases, foot_phase, duty_ratio) for foot_phase in foot_phases
    ])
    
    return h_steps

@dataclass
class State:
    """
    compressing the original 'state' and 'pipeline_state' into one dataclass
    Attributes:
        q: (q_size,) joint position vector
        qd: (qd_size,) joint velocity vector
        x: (num_links,) link position in world frame
        xd: (num_links,) link velocity in world frame
        contact: calculated contacts
    """
    # q: torch.Tensor
    # qd: torch.Tensor
    # x: torch.Tensor # not really needed rn
    # xd: torch.Tensor

    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: Dict[str, Any] 

#   contact: Optional[Contact]

@dataclass
class UnitreeH1WalkEnvConfig:
    task_name: str = "default"
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint.
    kp: float = 30.0
    # D gain, or a list of D gains for each joint.
    kd: float = 1.0
    debug: bool = False
    # dt of the environment step, not the underlying simulator step.
    dt: float = 0.02
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.02
    backend: str = "mjx"  # backend of the environment.
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    action_scale: float = 1.0  # scale of the action space.

    kp = np.array([
        200.0,
        200.0,
        200.0,  # left hips
        200.0,
        60.0,  # left knee, ankle
        200.0,
        200.0,
        200.0,  # right hips
        200.0,
        60.0,  # right knee, ankle
        200.0,  # torso
        60.0,
        60.0,
        60.0,
        60.0,  # left shoulder, elbow
        60.0,
        60.0,
        60.0,
        60.0,  # right shoulder, elbow
    ])
    kd = np.array([
        5.0,
        5.0,
        5.0,  # left hips
        5.0,
        1.5,  # left knee, ankle
        5.0,
        5.0,
        5.0,  # right hips
        5.0,
        1.5,  # right knee, ankle
        5.0,  # torso
        1.5,
        1.5,
        1.5,
        1.5,  # left shoulder, elbow
        1.5,
        1.5,
        1.5,
        1.5,  # right shoulder, elbow
    ])

    qpos = np.array("""
    0 0 0.98
    1 0 0 0
    0 0 -0.4 0.8 -0.4
    0 0 -0.4 0.8 -0.4
    0""".split()).astype(float)

    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "jog"
    ctrlrange = {
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

class H1GenesisEnv:
    def __init__(
            self, 
            num_envs: int,
            device: torch.device,
            rng: torch.Generator,
            config: UnitreeH1WalkEnvConfig,
            xml_path="/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/h1_real_feet.xml",
            show_viewer=False,
            show_fps=False,
        ):
        self.device = device    
        self.num_envs = num_envs
        self._config = config
        self.dt = config.dt
        n_frames = int(config.dt / config.timestep)

        # gs.init(backend=gs.gpu)
        scene = gs.Scene(
            show_viewer    = show_viewer,
            show_FPS       = show_fps,
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (3.5, -1.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
                res           = (512, 512), 
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=5),
            rigid_options = gs.options.RigidOptions(
                dt                = config.dt,
                constraint_solver = gs.constraint_solver.Newton, # Newton solver is faster than the default conjugate gradient (CG) solver.
            ),
        )
        plane = scene.add_entity(gs.morphs.Plane())
        self.entity = scene.add_entity(
            gs.morphs.MJCF(file=xml_path),
        )

        all_joints = self.entity.joints  
        self.actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        self.actuated_dof_names = [joint.name for joint in self.actuated_joints]
        self.actuated_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints]
        self.action_size = len(self.actuated_joints)

        # set ctrlrange because the mjcf parser is buggy 
        force_range = np.zeros((self.action_size, 2))
        for i, jname in enumerate(self.actuated_dof_names):
            if jname in config.ctrlrange:
                force_range[i] = config.ctrlrange[jname]
        force_range = torch.tensor(force_range, device=self.device) 

        # build first!
        self.scene = scene
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=(2.0, 2.0),
            n_envs_per_row=None,
        )
        self.entity.set_dofs_force_range(
            force_range[:, 0], force_range[:, 1], dofs_idx_local=self.actuated_dof_idxs
        )
        self.force_range = force_range  
        self.joint_torque_range = force_range  # NOTE that entity.get_dofs_force_range() returns 25-dim and first 6 is free joint


        self._pelvis_idx = self._find_body_idx("pelvis")
        self._torso_idx = self._find_body_idx("torso_link")
        self._left_foot_idx = self._find_body_idx("left_ankle_link") # this was left_foot site in xml
        self._right_foot_idx = self._find_body_idx("right_ankle_link") # this was right_foot site in xml
        
        self._gait = config.gait
        self._gait_phase = {
            "stand": np.zeros(2),
            "slow_walk": np.array([0.0, 0.5]),
            "walk": np.array([0.0, 0.5]),
            "jog": np.array([0.0, 0.5]),
        }
        # change to tensor
        self._gait_phase = {k: torch.tensor(v, device=self.device) for k, v in self._gait_phase.items()}

        self._gait_params = { # ratio, cadence, amplitude
            "stand": np.array([1.0, 1.0, 0.0]),
            "slow_walk": np.array([0.6, 0.8, 0.15]),
            "walk": np.array([0.5, 1.0, 0.15]),
            "jog": np.array([0.3, 2, 0.2]),
        }
        self._gait_params = {k: torch.tensor(v, device=self.device) for k, v in self._gait_params.items()}


        # joint limits and initial pose
        self._init_q = torch.tensor(config.qpos, device=self.device)
        self._default_pose = np.array(config.qpos[7:])
        # joint sampling range
        self.joint_range = torch.tensor(
            [
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-1.0, 1.0],
                [0.0, 1.74],
                [-0.6, 0.4],

                [-0.3, 0.3],
                [-0.3, 0.3],
                [-1.0, 1.0],
                [0.0, 1.74],
                [-0.6, 0.4],

                [-0.5, 0.5],

                [-0.78, 0.78],
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.3, 0.3],

                [-0.78, 0.78],
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.3, 0.3],
            ], device=self.device
        ) # this is smaller than xml joint range 
        # self._nv = self.entity.nv
        # self._nq = self.entity.nq
        # self._nq = len(self._init_q)
        self.rng = rng
        
        self.init_value_buffers()
        self.default_lin_vel_tar = torch.tensor([config.default_vx, config.default_vy, 0.0], device=self.device).repeat(self.num_envs, 1)
        # repeat for each env, so shape (num_envs, 3) 
        self.default_ang_vel_tar = torch.tensor([0.0, 0.0, config.default_vyaw], device=self.device).repeat(self.num_envs, 1)

        self.kp = torch.tensor(config.kp, device=self.device)
        self.kd = torch.tensor(config.kd, device=self.device)
 
    def _find_body_idx(self, name):
        idx = None 
        for link in self.entity.links:
            if link.name == name:
                idx = link.idx 
                break
        return idx

    def init_value_buffers(self):
        self.dof_qpos = torch.zeros((self.num_envs, self.action_size), device=self.device) # 19 joints
        self.dof_qvel = torch.zeros((self.num_envs, self.action_size), device=self.device)
        self.dof_ctrl = torch.zeros((self.num_envs, self.action_size), device=self.device) 

        self.torso_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.torso_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.torso_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.torso_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.pos_tar = torch.zeros((self.num_envs, 3), device=self.device)
        self.vel_tar = torch.zeros((self.num_envs, 3), device=self.device)
        self.ang_vel_tar = torch.zeros((self.num_envs, 3), device=self.device)
        self.yaw_tar = torch.zeros(self.num_envs, device=self.device)
        self.episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.z_feet = torch.zeros((self.num_envs, 2), device=self.device)
        self.z_feet_tar = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_contact = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device)

    def update_value_buffers(self):
        self.dof_qpos[:] = self.entity.get_dofs_position(self.actuated_dof_idxs) 
        self.dof_qvel[:] = self.entity.get_dofs_velocity(self.actuated_dof_idxs)
        self.torso_pos[:] = self.entity.get_links_pos()[:, self._torso_idx, :]
        self.torso_quat[:] = self.entity.get_links_quat()[:, self._torso_idx, :]
        self.torso_lin_vel[:] = self.entity.get_links_vel()[:, self._torso_idx, :]
        self.torso_ang_vel[:] = self.entity.get_links_ang()[:, self._torso_idx, :]

    def get_state(self, episode_step: int = 0) -> State:
        entity = self.entity 
        info = {
            "rng": self.rng,
            "pos_tar": self.pos_tar,
            "vel_tar": self.vel_tar,
            "yaw_tar": self.yaw_tar,
            "step": self.episode_step,
            "z_feet": self.z_feet,
            "z_feet_tar": self.z_feet_tar,
            "randomize_target": self._config.randomize_tasks,
            "last_contact": self.last_contact,
            "feet_air_time": self.feet_air_time,
        }
        obs = self._get_obs()
        state = State( 
            qpos=self.dof_qpos,
            qvel=self.dof_qvel,
            ctrl=self.dof_ctrl,
            obs=obs,
            reward=self.rew_buf,
            done=self.done_buf,
            info=info,
        )
        return state 
    
    def reset_idx(self, env_idxs=None):
        if env_idxs is None:
            env_idxs = torch.arange(self.num_envs)
        self.dof_qpos[env_idxs, :] = self._init_q
        self.qvel[env_idxs, :] = 0.0
        self.entity.set_dofs_position(
            position=self.qpos[env_idxs, :],
            dofs_idx_local=self.actuated_dof_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        ) 
        self.episode_step[env_idxs] = 0
        self.rew_buf[env_idxs] = 0.0
        self.done_buf[env_idxs] = False

        self.pos_tar[env_idxs, :] = torch.tensor([0.0, 0.0, 1.3], device=self.device)
        self.vel_tar[env_idxs, :] = 0.0 
        self.ang_vel_tar[env_idxs, :] = 0.0
        self.yaw_tar[env_idxs] = 0.0
        self.z_feet[env_idxs, :] = 0.0
        self.z_feet_tar[env_idxs, :] = 0.0
        self.last_contact[env_idxs, :] = False
        self.feet_air_time[env_idxs, :] = 0.0

        self.update_value_buffers()
        state = self.get_state()
        return state
    
    def _get_obs(self):
        # TODO: transform from global to body frame?
        vb = self.torso_lin_vel
        ab = self.torso_ang_vel
        obs = torch.cat([
            self.vel_tar,
            self.ang_vel_tar,
            self.dof_ctrl,
            self.dof_qpos,
            vb,
            ab,
            self.dof_qvel,
            ],
            dim=1
            )
        return obs

    def act2joint(self, act):
        act_normalized = (act * self._config.action_scale + 1.0) / 2.0 # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (self.joint_range[:, 1] - self.joint_range[:, 0])
        # joint_targets = torch.clamp(joint_targets, self.physical_joint_range[:, 0], self.physical_joint_range[:, 1])
        return joint_targets

    def act2tau(self, act):
        joint_target = self.act2joint(act)
        q = self.dof_qpos
        qd = self.dof_qvel
        q_err = joint_target - q
        tau = self.kp * q_err - self.kd * qd
        tau = torch.clamp(tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1])
        return tau
    
    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        state = self.get_state()
        return state
    
    # def sample_command(self): NOTE: default Not randomizing 
    #     # sample a random command in batched
    #     lin_vel_x = [-1.5, 1.5]  # min max [m/s]
    #     lin_vel_y = [-0.5, 0.5]  # min max [m/s]
    #     ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
    #     lin_vel_x = torch.random.uniform(lin_vel_x[0], lin_vel_x[1], generator=self.rng)    
    #     lin_vel_y = torch.random.uniform(lin_vel_y[0], lin_vel_y[1], generator=self.rng)
    #     ang_vel_yaw = torch.random.uniform(ang_vel_yaw[0], ang_vel_yaw[1], generator=self.rng)
    #     new_lin_vel_cmd = torch.tensor([lin_vel_x[0], lin_vel_y[1]], device=self.device)
    #     new_ang_vel_cmd = torch.tensor([0.0, 0.0, ang_vel_yaw], device=self.device)
    #     return new_lin_vel_cmd, new_ang_vel_cmd

    def step(self, state, action):
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "torque":
            ctrl = self.act2tau(action)
            self.entity.control_dofs_force(ctrl, self.actuated_dof_idxs)
        elif self._config.leg_control == "position":
            ctrl = joint_targets
            self.entity.control_dofs_position(joint_targets, self.actuated_dof_idxs)
        self.dof_ctrl[:] = ctrl
        
        self.scene.step()
        self.update_value_buffers() 
        # need broadcast 
        vel_tar = self.default_lin_vel_tar * self.episode_step[:, None] * self.dt / self._config.ramp_up_time
        self.vel_tar[:] = torch.min(vel_tar, self.default_lin_vel_tar) # broadcasting
        
        ang_tar = self.default_ang_vel_tar * self.episode_step[:, None] * self.dt / self._config.ramp_up_time
        self.ang_vel_tar[:] = torch.min(ang_tar, self.default_ang_vel_tar)
        
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(duty_ratio, cadence, amplitude, phases, self.episode_step * self.dt) # shape (2, num_envs)
        z_feet_tar = z_feet_tar.T # shape (num_envs, 2) 
        foot_pos = self.entity.get_links_pos()[:, [self._left_foot_idx, self._right_foot_idx], :]
 
        z_feet = foot_pos[:, :, 2]
        # TODO: re-implement the contact approximation 
        
        reward_gaits = -1 * torch.sum((z_feet_tar - z_feet) ** 2, dim=-1)
        
        foot_contact_z = foot_pos[:, :, 2]
        contact = foot_contact_z < 1e-3 
        contact_filt_mm = contact | self.last_contact 
        first_contact = (self.feet_air_time > 0) * contact_filt_mm
        self.feet_air_time += self.dt
        reward_air_time = torch.sum(
            (self.feet_air_time - 0.1) * first_contact, dim=-1
        )

        pos_tar = (
            self.pos_tar + self.vel_tar * self.dt * self.episode_step[:, None]
        )
        reward_pos = -torch.sum((self.torso_pos - pos_tar) ** 2, dim=-1)

        # vec_tar = torc
        reward_upright = torch.zeros(self.num_envs, device=self.device)

        # yaw orientation reward
        yaw_tar = self.yaw_tar + self.ang_vel_tar[:, 2] * self.dt * self.episode_step
        yaw = quat_to_xyz(self.torso_quat)[:, 2]
        d_yaw = yaw - yaw_tar
        # reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw))) -> rewrite in torch:
        reward_yaw = -1 * torch.square(torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw)))

        vb = inv_transform_by_quat(
            self.torso_lin_vel, self.torso_quat
        )
        ab = inv_transform_by_quat(
            self.torso_ang_vel, self.torso_quat
        ) 
        reward_vel = -1 * torch.sum((vb[:, :2] - self.vel_tar[:, :2]) ** 2, dim=-1)
        reward_ang_vel = -1 * (ab[:, 2] - self.ang_vel_tar[:, 2]) ** 2 

        # height reward 
        reward_height = -1 * (self.torso_pos[:, 2] - self.pos_tar[:, 2]) ** 2
        

        # foot level reward 
        reward_foot_level = torch.zeros(self.num_envs, device=self.device) # TODO 

        reward_energy = -1 * torch.sum(
            (ctrl / self.force_range[:, 1] * self.dof_qvel / 160.0 ) ** 2, 
            dim=-1
        )
        reward_alive = 1.0 - self.done_buf.float()
        
        # reward
        reward = (
            reward_gaits * 10.0
            + reward_air_time * 0.0
            + reward_pos * 0.0
            + reward_upright * 0.5
            + reward_yaw * 0.5
            # + reward_pose * 0.0
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 0.5
            + reward_foot_level * 0.02
            + reward_energy * 0.01
            + reward_alive * 0.0
        ) 

        # done
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)[None, :]
        up_trans = transform_by_quat(up, self.torso_quat)
        done = torch.einsum("ij,ij->i", up, up_trans) < 0
        done |= torch.any(self.dof_qpos < self.joint_range[:, 0])
        done |= torch.any(self.dof_qpos > self.joint_range[:, 1])
        done |= self.torso_pos[:, 2] < 0.18
         
        self.done_buf[:] = done
        self.rew_buf[:] = reward
        
        self.episode_step += 1
        self.z_feet[:] = z_feet
        self.z_feet_tar[:] = z_feet_tar
        self.last_contact[:] = contact
        
        state = self.get_state()
        return state


# if __name__ == "__main__":
#     gs.init(backend=gs.gpu)
#     env = H1GenesisEnv(
#         num_envs=2,
#         device=torch.device("cuda"),
#         rng=torch.Generator(),
#         config=UnitreeH1WalkEnvConfig(),
#         show_viewer=True,
#     )
#     # obs = env._get_obs()
#     state = env.reset()
#     while True:
#         action = torch.randn(19, device=torch.device("cuda"))
#         state = env.step(state, action) 