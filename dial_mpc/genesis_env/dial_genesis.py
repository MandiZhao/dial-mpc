import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time 
import yaml
from tqdm import tqdm
import torch
import numpy as np
import genesis as gs
from scipy.interpolate import InterpolatedUnivariateSpline
import torch.nn.functional as F

from utils.io_utils import get_example_path, load_dataclass_from_dict
# from dial_mpc.examples import examples
from core.dial_config import DialConfig
from genesis_env.genesis_h1_env import UnitreeH1WalkEnvConfig, H1GenesisEnv

class InterpolationModel:
    def __init__(self, step_nodes: torch.Tensor, step_us: torch.Tensor):
        self.step_nodes = step_nodes
        self.step_us = step_us

    def node2u(self, nodes):
        # nodes = torch.tensor(nodes, dtype=torch.float32)
        nodes = nodes.cpu().numpy() if isinstance(nodes, torch.Tensor) else nodes        
        # Use numpy/scipy for spline interpolation 
        spline = InterpolatedUnivariateSpline(self.step_nodes.cpu().numpy(), nodes, k=2)
        us = spline(self.step_us.cpu().numpy()) 
        # Convert back to PyTorch tensor
        return torch.tensor(us, dtype=torch.float32)
    
    def node2u_vmap(self, nodes):
        # Vectorized over horizon (axis 1)
        return torch.stack([self.node2u(nodes[:, i]) for i in range(nodes.shape[1])], dim=1)

    def node2u_vvmap(self, batch_nodes):
        # Vectorized over batch (axis 0)
        return torch.stack([self.node2u_vmap(nodes) for nodes in batch_nodes], dim=0)
    
    def u2node(self, us):
        """
        Interpolates `us` (defined on `step_us`) back to `step_nodes` using a cubic spline (k=2).
        """
        # us = torch.tensor(us, dtype=torch.float32)
        us = us.cpu().numpy() if isinstance(us, torch.Tensor) else us
        # Perform inverse spline interpolation using scipy
        spline = InterpolatedUnivariateSpline(self.step_us.cpu().numpy(), us, k=2)
        nodes = spline(self.step_nodes.cpu().numpy())
        
        # Convert result back to PyTorch tensor
        return torch.tensor(nodes, dtype=torch.float32)

    def u2node_vmap(self, us):
        # Vectorized over horizon (axis 1)
        return torch.stack([self.u2node(us[:, i]) for i in range(us.shape[1])], dim=1)
      
def softmax_update_gs(weights, Y0s, sigma, mu_0t): 
    mu_0tm1 = torch.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma

def rollout_us_fn(env, state, us):
    state = env.step(state, us)
    return (state.reward, state.pipeline_state)
# NOTE: remove the need for pipeline_state
def rollout_us_scan(env, state, us):
    # us: shape (num_envs, Hnode + 1, nu)
    # should return rewss -> (num_envs, Hnode + 1)
    rewss = []
    for h in range(us.shape[1]):
        state = env.step(state, us[:, h])
        rewss.append(state.reward)
    rewss = torch.stack(rewss, dim=1)
    return rewss


class MBDPIGenesis:
    def __init__(self, args, env, device=torch.device("cuda")):
        self.args = args
        self.env = env 
        self.nu = env.action_size
        self.device = device
        self.update_fn = softmax_update_gs
        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = np.log(sigma1/sigma0) / args.Ndiffuse
        sigmas = A * np.exp(B * np.arange(args.Ndiffuse))
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.sigma_control = (
            args.horizon_diffuse_factor ** np.arange(args.Hnode + 1)[::-1]
        )
        self.sigma_control = torch.tensor(self.sigma_control, dtype=torch.float32)

        self.ctrl_dt = 0.02
        self.step_us = np.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_us = torch.tensor(self.step_us, dtype=torch.float32)

        self.step_nodes = np.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.step_nodes = torch.tensor(self.step_nodes, dtype=torch.float32)

        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        self.rollout_us = rollout_us_fn
        self.rollout_us_scan = rollout_us_scan
        self.node2u = InterpolationModel(self.step_nodes, self.step_us).node2u
        self.node2u_vmap = InterpolationModel(self.step_nodes, self.step_us).node2u_vmap
        self.node2u_vvmap = InterpolationModel(self.step_nodes, self.step_us).node2u_vvmap

        self.u2node = InterpolationModel(self.step_nodes, self.step_us).u2node
        self.u2node_vmap = InterpolationModel(self.step_nodes, self.step_us).u2node_vmap

    def reverse_once(self, state, seed, Ybar_i, noise_scale):
        # Sample from q_i
        Y0s_rng = torch.Generator()
        Y0s_rng.manual_seed(seed)
        eps_Y = torch.randn((self.args.Nsample, self.args.Hnode + 1, self.nu), generator=Y0s_rng, dtype=torch.float32)
        eps_Y = eps_Y.to(self.device)
        noise_scale = noise_scale.to(self.device)
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        Y0s = Y0s.to(self.device)
        
        # Ensure the first control is fixed
        Y0s[:, 0] = Ybar_i[0] 
        # Append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = torch.cat([Y0s, Ybar_i[None]], dim=0)
        Y0s = torch.clamp(Y0s, -1.0, 1.0)
        
        # Convert Y0s to us
        us = self.node2u_vvmap(Y0s)
        us = us.to(self.device)
        # Estimate mu_0tm1
        # rewss, pipeline_statess = self.rollout_us(state, us)
        rewss = self.rollout_us_scan(self.env, state, us)
        rew_Ybar_i = rewss[-1].mean()
    
        # qss = pipeline_statess["q"]
        # qdss = pipeline_statess["qd"]
        # xss = pipeline_statess["x"]["pos"]
        rews = rewss.mean(dim=-1)
        
        logp0 = (rews - rew_Ybar_i) / rews.std(dim=-1) / self.args.temp_sample
        
        weights = F.softmax(logp0, dim=0) 
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)
        
        # Update only with reward
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)
        # qbar = torch.einsum("n,nij->ij", weights, qss)
        # qdbar = torch.einsum("n,nij->ij", weights, qdss)
        # xbar = torch.einsum("n,nijk->ijk", weights, xss)
        
        info = {
            "rews": rews,
            # "qbar": qbar,
            # "qdbar": qdbar,
            # "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }
        
        return torch.randint(0, 2**32 - 1, (1,)).item(), Ybar, info

    def reverse(self, state, YN, seed):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, info = self.reverse_once(
                    state, seed, Yi, self.sigmas[i] * torch.ones(self.args.Hnode + 1)
                )
                # Simulating `block_until_ready` with PyTorch synchronization (if using GPU)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{info['rews'].mean().item():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    def shift(self, Y):
        # Convert nodes to controls
        device = Y.device
        u = self.node2u_vmap(Y)
        
        # Roll controls backward and set the last control to zero
        u = torch.roll(u, shifts=-1, dims=0)
        u[-1] = torch.zeros(self.nu)
        
        # Convert controls back to nodes
        Y = self.u2node_vmap(u)
        return Y.to(device)

    def shift_Y_from_u(self, u, n_step):
        # Roll controls backward by n_step
        u = torch.roll(u, shifts=-n_step, dims=0)
        
        # Set the last n_step controls to zero
        u[-n_step:] = torch.zeros_like(u[-n_step:])
        
        # Convert controls back to nodes
        Y = self.u2node_vmap(u)
        return Y

def main():

    def reverse_scan(seed, Y0, state, mbdpi, noise_scale):
        Y0s, infos = [], []
        n_noise = noise_scale.shape[0]
        for i in range(n_noise): 
            seed, Y0, info = mbdpi.reverse_once(state, seed, Y0, noise_scale[i])
            Y0s.append(Y0)
            infos.append(info)
        return (rng, Y0, state), info

    config_fname = "/home/mandiz/dial-mpc/dial_mpc/examples/unitree_h1_jog.yaml"
    config_dict = yaml.safe_load(open(config_fname, "r"))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = torch.Generator()
    rng.manual_seed(dial_config.seed)
    rng_reset = torch.Generator()
    rng_reset.manual_seed(dial_config.seed + 1)

    device = torch.device("cuda")
    gs.init(backend=gs.gpu)
    env = H1GenesisEnv(
        xml_path="/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/h1_real_feet.xml",
        # xml_path="/home/mandiz/dial-mpc/dial_mpc/models/unitree_h1/mjx_h1_walk_real_feet.xml",
        num_envs=dial_config.Nsample + 1,
        device=device,
        rng=torch.Generator(),
        config=UnitreeH1WalkEnvConfig(),
        show_viewer=True,
        show_fps=False,
    )
    mbdpi = MBDPIGenesis(dial_config, env)
    
    state_init = env.reset(rng_reset)
    YN = np.zeros([dial_config.Hnode + 1, mbdpi.nu])
    YN = torch.tensor(YN, dtype=torch.float32, device=device)
    Y0 = YN

    Nstep = dial_config.n_steps 
    seed = 42
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    traj_diffuse_factors_init = (
        mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (np.arange(dial_config.Ndiffuse_init))[:, None]
    )
    traj_diffuse_factors_init = torch.tensor(traj_diffuse_factors_init, dtype=torch.float32, device=device)

    traj_diffuse_factors = (
        mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (np.arange(dial_config.Ndiffuse))[:, None]
    )
    traj_diffuse_factors = torch.tensor(traj_diffuse_factors, dtype=torch.float32, device=device)

    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            state = env.step(state, Y0[0])
            rollout.append(state)
            rews.append(state.reward) 
            us.append(Y0[0])

            Y0 = mbdpi.shift(Y0)
            n_diffuse = dial_config.Ndiffuse # 4
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init # 10
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            _factors = traj_diffuse_factors_init if t == 0 else traj_diffuse_factors
            # original did a jax.lax.scan here to iterate over the first dim of traj_diffuse_factors:
            # (rng, Y0, _), info = jax.lax.scan(
            #     reverse_scan, (rng, Y0, state), traj_diffuse_factors
            # )
            (rng, Y0, _), info = reverse_scan(seed, Y0, state, mbdpi,_factors)
            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            avg_rew = state.reward.mean().item()
            print("t: ", t, "rew: ", avg_rew, "freq: ", freq)
            pbar.set_postfix({"rew": f"{avg_rew:.2e}", "freq": f"{freq:.2f}"})
    
    rew = np.array(rews).mean()
    print(f"Final mean reward: {rew:.2e}")

if __name__ == "__main__":
    main()