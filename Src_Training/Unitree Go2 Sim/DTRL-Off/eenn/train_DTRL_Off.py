"""Go2 three-port combined SAC (e1 / e2 / efull), Genesis parallel environment.

- e1: obs → 512 → out
- e2: obs → 512 → 256 → 128 → out
- efull: obs → 512 → 256 → 128 → 128 → 128 → out
- The first ``warmup_steps`` (default 20M) only efull backtransmission; later combined with e1/e2/efull
- 0~total records the entire efull training curve (``train/episode_return_efull``)

Default total_timesteps=50_000_896 (consistent with ``run_flashsac_go2.py``), warmup_steps=20M.

Run::

    cd ~/codespace/RTSS
    python unitree_go2/DTRL-Off/eenn/joint_train.py"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_OFFPOLICY_DIR = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_OFFPOLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFPOLICY_DIR))

from action_utils import BoxSpace, random_scaled_action_batch, unscale_action_batch
from eenn_network import ActorEENN3Go2, CriticDeepTwin, soft_update

_GENESIS_LOCOMOTION_REL = Path("Genesis") / "examples" / "locomotion"
_DEFAULT_RUNS_ROOT = _OFFPOLICY_DIR / "runs"
_DEFAULT_TOTAL_STEPS = 50_000_896
_DEFAULT_WARMUP_STEPS = 20_000_000


def _z(g: torch.Tensor | None, p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p) if g is None else g


def assign_triple_joint_grads_go2(
    shared: list[torch.nn.Parameter],
    fc2: list[torch.nn.Parameter],
    fc3: list[torch.nn.Parameter],
    deep: list[torch.nn.Parameter],
    e1: list[torch.nn.Parameter],
    e2: list[torch.nn.Parameter],
    e3: list[torch.nn.Parameter],
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    loss3: torch.Tensor,
    w1: float,
    w2: float,
    w3: float,
) -> None:
    g1_e1 = torch.autograd.grad(loss1, e1, retain_graph=True, allow_unused=True)
    g1_s = torch.autograd.grad(loss1, shared, retain_graph=True, allow_unused=True)

    g2_e2 = torch.autograd.grad(loss2, e2, retain_graph=True, allow_unused=True)
    g2_f3 = torch.autograd.grad(loss2, fc3, retain_graph=True, allow_unused=True)
    g2_f2 = torch.autograd.grad(loss2, fc2, retain_graph=True, allow_unused=True)
    g2_s = torch.autograd.grad(loss2, shared, retain_graph=True, allow_unused=True)

    g3_e3 = torch.autograd.grad(loss3, e3, retain_graph=True, allow_unused=True)
    g3_d = torch.autograd.grad(loss3, deep, retain_graph=True, allow_unused=True)
    g3_f3 = torch.autograd.grad(loss3, fc3, retain_graph=True, allow_unused=True)
    g3_f2 = torch.autograd.grad(loss3, fc2, retain_graph=True, allow_unused=True)
    g3_s = torch.autograd.grad(loss3, shared, retain_graph=False, allow_unused=True)

    for p, a, b, c in zip(shared, g1_s, g2_s, g3_s):
        p.grad = w1 * _z(a, p) + w2 * _z(b, p) + w3 * _z(c, p)
    for p, b, c in zip(fc2, g2_f2, g3_f2):
        p.grad = w2 * _z(b, p) + w3 * _z(c, p)
    for p, b, c in zip(fc3, g2_f3, g3_f3):
        p.grad = w2 * _z(b, p) + w3 * _z(c, p)
    for p, g in zip(deep, g3_d):
        p.grad = w3 * _z(g, p)
    for p, g in zip(e1, g1_e1):
        p.grad = w1 * _z(g, p)
    for p, g in zip(e2, g2_e2):
        p.grad = w2 * _z(g, p)
    for p, g in zip(e3, g3_e3):
        p.grad = w3 * _z(g, p)


def _resolve_locomotion_dir(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_loc = os.environ.get("GENESIS_LOCOMOTION")
    if env_loc:
        candidates.append(Path(env_loc).expanduser())
    rtss_root = os.environ.get("RTSS_ROOT")
    if rtss_root:
        candidates.append(Path(rtss_root).expanduser() / _GENESIS_LOCOMOTION_REL)

    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        candidates.append(base / _GENESIS_LOCOMOTION_REL)

    tried: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        p = raw.resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if p.is_dir():
            return p

    raise FileNotFoundError(
        "Genesis locomotion not found. Tried:\n  "
        + "\n  ".join(tried[:12])
        + ("\n  ..." if len(tried) > 12 else "")
    )


def _setup_go2_import(locomotion_dir: Path) -> None:
    loc = str(locomotion_dir.resolve())
    if loc not in sys.path:
        sys.path.insert(0, loc)


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def make_action_space(clip: float, act_dim: int) -> BoxSpace:
    return BoxSpace.symmetric(clip, act_dim)


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_vec(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        for i in range(n):
            idx = self.ptr
            self.obs[idx] = obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_obs[idx] = next_obs[i]
            self.dones[idx] = dones[i]
            self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=device)
        act = torch.as_tensor(self.actions[idx], device=device)
        rew = torch.as_tensor(self.rewards[idx], device=device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=device)
        done = torch.as_tensor(self.dones[idx], device=device)
        return obs, act, rew, next_obs, done


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_exit_episode_go2(
    env,
    actor: ActorEENN3Go2,
    device: torch.device,
    action_space: BoxSpace,
    *,
    exit_id: int,
) -> tuple[float, int]:
    """Evaluates a single round when ``gs.init`` has been created and ``Go2Env(num_envs=1)`` has been created."""
    import genesis as gs

    obs_dict = env.reset()
    obs = obs_dict["policy"][0].cpu().numpy()
    ep_ret = 0.0
    ep_len = 0
    max_steps = env.max_episode_length + 50
    for _ in range(max_steps):
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        m1, _ls1, m2, _ls2, m3, _ls3 = actor.forward(o)
        mean = m1 if exit_id == 1 else (m2 if exit_id == 2 else m3)
        a_scaled = torch.tanh(mean).cpu().numpy().reshape(-1)
        action_env = unscale_action_batch(a_scaled.reshape(1, -1), action_space)[0]
        act_t = torch.as_tensor(action_env, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
        obs_dict, rews, dones, _extras = env.step(act_t)
        obs = obs_dict["policy"][0].cpu().numpy()
        ep_ret += float(rews[0].item())
        ep_len += 1
        if bool(dones[0].item()):
            break
    return ep_ret, ep_len


def eval_three_exit_weighted_return(
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    actor: ActorEENN3Go2,
    device: torch.device,
    action_space: BoxSpace,
    *,
    start_seed: int,
    n_episodes: int,
    w1: float,
    w2: float,
    w3: float,
) -> tuple[float, float, float, float]:
    """Evaluated in a stand-alone Genesis session (will destroy the current gs, the caller is responsible for restoring the training environment)."""
    import genesis as gs
    from go2_env import Go2Env

    try:
        gs.destroy()
    except Exception:
        pass

    gs.init(
        backend=gs.gpu if device.type == "cuda" else gs.cpu,
        precision="32",
        logging_level="warning",
        seed=start_seed,
        performance_mode=True,
    )
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    rets1, rets2, rets3 = [], [], []
    for _ep in range(n_episodes):
        rets1.append(run_exit_episode_go2(env, actor, device, action_space, exit_id=1)[0])
        rets2.append(run_exit_episode_go2(env, actor, device, action_space, exit_id=2)[0])
        rets3.append(run_exit_episode_go2(env, actor, device, action_space, exit_id=3)[0])
    try:
        gs.destroy()
    except Exception:
        pass

    r1 = float(np.mean(rets1)) if rets1 else 0.0
    r2 = float(np.mean(rets2)) if rets2 else 0.0
    r3 = float(np.mean(rets3)) if rets3 else 0.0
    return r1, r2, r3, w1 * r1 + w2 * r2 + w3 * r3


def reinit_training_env(
    *,
    num_envs: int,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    device: torch.device,
    seed: int,
) -> tuple:
    """eval will ``gs.destroy``, and the parallel training environment needs to be rebuilt."""
    import genesis as gs
    from go2_env import Go2Env

    gs.init(
        backend=gs.gpu if device.type == "cuda" else gs.cpu,
        precision="32",
        logging_level="warning",
        seed=seed,
        performance_mode=True,
    )
    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    obs_dict = env.reset()
    obs = obs_dict["policy"].detach().cpu().numpy()
    return env, obs


def save_checkpoint(
    path: str,
    *,
    step: int,
    actor: ActorEENN3Go2,
    critic: CriticDeepTwin,
    critic_target: CriticDeepTwin,
    opt_a: optim.Optimizer,
    opt_c: optim.Optimizer,
    log_alpha: torch.Tensor,
    extra: dict | None = None,
) -> None:
    payload = {
        "step": step,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "opt_a": opt_a.state_dict(),
        "opt_c": opt_c.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 three-port joint SAC (e1/e2/efull)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-timesteps", type=int, default=_DEFAULT_TOTAL_STEPS)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=_DEFAULT_WARMUP_STEPS,
        help="Previously, actor only efull(exit3) backtransmission (default 20M)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--w-exit1", type=float, default=0.2, dest="w_exit1")
    p.add_argument("--w-exit2", type=float, default=0.3, dest="w_exit2")
    p.add_argument("--w-exit3", type=float, default=0.5, dest="w_exit3")
    p.add_argument("--runs-root", type=str, default=str(_DEFAULT_RUNS_ROOT))
    p.add_argument(
        "--save-freq",
        type=int,
        default=max(1, _DEFAULT_TOTAL_STEPS // 5),
        help="Consistent with FlashSAC: save ckpt approximately every 20%% of total steps",
    )
    p.add_argument("--eval-freq", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-start-seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10_000)
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--no-split-joint-grad", action="store_true")
    p.add_argument("--genesis-locomotion", type=str, default=None)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    locomotion_dir = _resolve_locomotion_dir(args.genesis_locomotion)
    _setup_go2_import(locomotion_dir)

    import genesis as gs
    from go2_env import Go2Env

    device = torch.device(args.device)
    set_seed(args.seed)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    act_dim = int(env_cfg["num_actions"])
    action_space = make_action_space(env_cfg["clip_actions"], act_dim)

    gs.init(
        backend=gs.gpu if device.type == "cuda" else gs.cpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    obs_dim = int(env.obs_buf.shape[-1])

    actor = ActorEENN3Go2(obs_dim, act_dim).to(device)
    critic = CriticDeepTwin(obs_dim, act_dim).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad = False

    opt_a = optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -float(act_dim)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_root, f"joint_sac_go2_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    best_model_dir = os.path.join(run_dir, "best_model")
    final_model_dir = os.path.join(run_dir, "final_model")
    for d in (ckpt_dir, tb_dir, best_model_dir, final_model_dir):
        os.makedirs(d, exist_ok=True)

    writer = SummaryWriter(tb_dir)
    best_weighted = -np.inf
    next_eval_at = args.warmup_steps
    next_save_at = args.save_freq

    w1, w2, w3 = args.w_exit1, args.w_exit2, args.w_exit3

    print(f"[INFO] Genesis locomotion: {locomotion_dir}")
    print(f"[INFO] total_timesteps={args.total_timesteps}")
    print(f"[INFO] warmup_steps={args.warmup_steps} (efull only)")
    print(f"[INFO] num_envs={args.num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"[INFO] save_freq={args.save_freq}, eval_freq={args.eval_freq} (eval after warmup)")
    print(f"[INFO] run_dir={run_dir}")

    obs_dict = env.reset()
    obs = obs_dict["policy"].detach().cpu().numpy()
    global_step = 0
    ep_return_efull = np.zeros(args.num_envs, dtype=np.float64)
    ep_len_efull = np.zeros(args.num_envs, dtype=np.int64)
    pbar = tqdm(
        total=args.total_timesteps,
        unit="step",
        desc="joint_sac_go2",
        dynamic_ncols=True,
        disable=args.no_progress_bar,
    )

    while global_step < args.total_timesteps:
        n = args.num_envs
        if global_step < args.learning_starts:
            buf_a, env_a = random_scaled_action_batch(action_space, n)
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _a1, _lp1, _a2, _lp2, a3, _lp3 = actor.sample_all(o)
                buf_a = a3.cpu().numpy()
            env_a = unscale_action_batch(buf_a, action_space)

        act_t = torch.as_tensor(env_a, dtype=gs.tc_float, device=gs.device)
        next_obs_dict, rews, dones, _extras = env.step(act_t)
        next_obs = next_obs_dict["policy"].detach().cpu().numpy()
        rewards_np = rews.detach().cpu().numpy().reshape(-1)
        dones_np = dones.detach().cpu().numpy().reshape(-1).astype(np.float32)

        ep_return_efull += rewards_np
        ep_len_efull += 1
        done_mask = dones_np > 0.5
        if np.any(done_mask):
            for i in np.where(done_mask)[0]:
                writer.add_scalar("train/episode_return_efull", float(ep_return_efull[i]), global_step)
                writer.add_scalar("train/episode_len_efull", float(ep_len_efull[i]), global_step)
                ep_return_efull[i] = 0.0
                ep_len_efull[i] = 0

        buffer.add_vec(obs, buf_a, rewards_np, next_obs, dones_np)
        obs = next_obs
        global_step += n
        pbar.update(n)

        do_train = buffer.size >= max(args.batch_size, args.learning_starts) and global_step >= args.learning_starts

        if do_train:
            alpha = log_alpha.exp().clamp(min=1e-8)
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(args.batch_size, device)

            with torch.no_grad():
                _a1n, _lp1n, _a2n, _lp2n, a3_next, log_pi3_next = actor.sample_all(b_next_obs)
                q1_t, q2_t = critic_target(b_next_obs, a3_next)
                q_min_t = torch.min(q1_t, q2_t)
                target_q = b_rew + (1.0 - b_done) * args.gamma * (q_min_t - alpha * log_pi3_next)

            q1, q2 = critic(b_obs, b_act)
            loss_c = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            a1, log_pi1, a2, log_pi2, a3, log_pi3 = actor.sample_all(b_obs)
            qm1 = torch.min(*critic(b_obs, a1))
            qm2 = torch.min(*critic(b_obs, a2))
            qm3 = torch.min(*critic(b_obs, a3))
            loss_pi_1 = (alpha * log_pi1 - qm1).mean()
            loss_pi_2 = (alpha * log_pi2 - qm2).mean()
            loss_pi_3 = (alpha * log_pi3 - qm3).mean()
            loss_pi = w1 * loss_pi_1 + w2 * loss_pi_2 + w3 * loss_pi_3

            opt_a.zero_grad()
            in_warmup = global_step < args.warmup_steps
            if in_warmup:
                loss_pi_3.backward()
            elif args.no_split_joint_grad:
                loss_pi.backward()
            else:
                sh, fc2, fc3, deep, e1, e2, e3 = actor.joint_param_groups()
                assign_triple_joint_grads_go2(
                    sh, fc2, fc3, deep, e1, e2, e3,
                    loss_pi_1, loss_pi_2, loss_pi_3, w1, w2, w3,
                )
            opt_a.step()

            loss_alpha = -(log_alpha * (log_pi3.detach() + target_entropy)).mean()
            opt_alpha.zero_grad()
            loss_alpha.backward()
            opt_alpha.step()

            soft_update(critic_target, critic, args.tau)

            if global_step % args.log_interval < n:
                writer.add_scalar("loss/critic", loss_c.item(), global_step)
                writer.add_scalar(
                    "loss/actor",
                    loss_pi_3.item() if in_warmup else loss_pi.item(),
                    global_step,
                )
                writer.add_scalar("loss/actor_exit1", loss_pi_1.item(), global_step)
                writer.add_scalar("loss/actor_exit2", loss_pi_2.item(), global_step)
                writer.add_scalar("loss/actor_exit3", loss_pi_3.item(), global_step)
                writer.add_scalar("warmup/active", 1.0 if in_warmup else 0.0, global_step)

        if global_step >= args.warmup_steps and global_step >= next_eval_at:
            while global_step >= next_eval_at:
                next_eval_at += args.eval_freq
            r1, r2, r3, weighted = eval_three_exit_weighted_return(
                env_cfg, obs_cfg, reward_cfg, command_cfg,
                actor, device, action_space,
                start_seed=args.eval_start_seed,
                n_episodes=args.eval_episodes,
                w1=w1, w2=w2, w3=w3,
            )
            writer.add_scalar("eval/mean_reward_e1", r1, global_step)
            writer.add_scalar("eval/mean_reward_e2", r2, global_step)
            writer.add_scalar("eval/mean_reward_efull", r3, global_step)
            writer.add_scalar("eval/weighted_mean_return", weighted, global_step)
            print(
                f"[Eval] step={global_step} e1={r1:.4f} e2={r2:.4f} efull={r3:.4f} "
                f"weighted={weighted:.4f}"
            )
            if weighted > best_weighted:
                best_weighted = weighted
                best_path = os.path.join(best_model_dir, "best_model.pt")
                save_checkpoint(
                    best_path,
                    step=global_step,
                    actor=actor,
                    critic=critic,
                    critic_target=critic_target,
                    opt_a=opt_a,
                    opt_c=opt_c,
                    log_alpha=log_alpha,
                    extra={
                        "mean_return_e1": r1,
                        "mean_return_e2": r2,
                        "mean_return_efull": r3,
                        "weighted_mean_return": weighted,
                        "warmup_steps": args.warmup_steps,
                    },
                )
                print(f"[Eval] new best weighted={weighted:.4f} -> {best_path}")

            env, obs = reinit_training_env(
                num_envs=args.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                device=device,
                seed=args.seed + global_step,
            )
            ep_return_efull.fill(0.0)
            ep_len_efull.fill(0)

        if global_step >= next_save_at:
            while global_step >= next_save_at:
                next_save_at += args.save_freq
            ckpt_path = os.path.join(ckpt_dir, f"joint_sac_{global_step}_steps.pt")
            save_checkpoint(
                ckpt_path,
                step=global_step,
                actor=actor,
                critic=critic,
                critic_target=critic_target,
                opt_a=opt_a,
                opt_c=opt_c,
                log_alpha=log_alpha,
                extra={
                    "warmup_steps": args.warmup_steps,
                    "w_exit1": w1,
                    "w_exit2": w2,
                    "w_exit3": w3,
                    "total_timesteps": args.total_timesteps,
                },
            )
            print(f"[INFO] checkpoint: {ckpt_path}")

    pbar.close()

    final_path = os.path.join(final_model_dir, "joint_sac_final.pt")
    save_checkpoint(
        final_path,
        step=global_step,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        opt_a=opt_a,
        opt_c=opt_c,
        log_alpha=log_alpha,
    )
    print(f"[INFO] final_model={final_path}")

    writer.close()
    try:
        gs.destroy()
    except Exception:
        pass


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
