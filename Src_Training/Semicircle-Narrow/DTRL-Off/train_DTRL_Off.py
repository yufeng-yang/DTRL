"""Semicircle v5 three-port combined SAC (e1/e2/efull).

- e1: obs → 256 → out
- e2: obs → 256 → 128 → 128 → out
- efull (exit3): obs → 256 → 128 → 128 → 64 → 64 → out
- Before ``--warmup-steps`` (default 1M) only efull backpropagation; after weighted combination e1/e2/efull
- 16 channels of parallel sampling; eval only after warmup is completed (three-port weighted return selects best)
- checkpoint / eval directory structure alignment ``b1_full_only/train_sac_full.py``

Run: conda activate sb3sg && python joint_train.py"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_SEMICIRCLE_NARROW = Path(__file__).resolve().parents[1]
if str(_SEMICIRCLE_NARROW) not in sys.path:
    sys.path.insert(0, str(_SEMICIRCLE_NARROW))

import Semicircle_env  # noqa: F401
import safety_gymnasium

from action_utils import random_scaled_action_batch, unscale_action_batch
from eenn_network import ActorEENN3Semicircle, CriticDeepTwin, soft_update


def _z(g: torch.Tensor | None, p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p) if g is None else g


def assign_triple_joint_grads_semicircle(
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
    """e1 only updates backbone+e1; e2 updates to fc3; efull updates the full depth segment."""
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


class SafetyToGymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        inner_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(inner_getter):
            return inner_getter(name)
        return getattr(self.env, name)


def make_env(env_id: str):
    def _init():
        env = safety_gymnasium.make(env_id, render_mode=None)
        env = SafetyToGymnasiumWrapper(env)
        env = Monitor(env)
        return env

    return _init


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

    def add(self, t: Transition) -> None:
        i = self.ptr
        self.obs[i] = t.obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.next_obs[i] = t.next_obs
        self.dones[i] = t.done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_vec(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for i in range(obs.shape[0]):
            self.add(
                Transition(
                    obs=obs[i],
                    action=actions[i],
                    reward=float(rewards[i]),
                    next_obs=next_obs[i],
                    done=float(dones[i]),
                )
            )

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


def run_exit_episode(
    env_id: str,
    actor: ActorEENN3Semicircle,
    device: torch.device,
    action_space: spaces.Box,
    *,
    seed: int,
    exit_id: int,
) -> tuple[float, int]:
    """Specify the exit to run a single round and return (undiscounted return, episode length)."""
    assert exit_id in (1, 2, 3)
    env = safety_gymnasium.make(env_id, render_mode=None)
    env = SafetyToGymnasiumWrapper(env)
    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len = 0
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m1, _ls1, m2, _ls2, m3, _ls3 = actor.forward(o)
            mean = m1 if exit_id == 1 else (m2 if exit_id == 2 else m3)
            a_scaled = torch.tanh(mean).cpu().numpy().reshape(-1)
            action_env = unscale_action_batch(a_scaled.reshape(1, -1), action_space)[0]
        obs, reward, terminated, truncated, _ = env.step(action_env)
        ep_ret += float(reward)
        ep_len += 1
    env.close()
    return ep_ret, ep_len


def eval_exit_mean_return(
    env_id: str,
    actor: ActorEENN3Semicircle,
    device: torch.device,
    action_space: spaces.Box,
    *,
    start_seed: int,
    n_episodes: int,
    exit_id: int,
) -> float:
    """Specifies exit (1=e1, 2=e2, 3=efull) deterministic mean action evaluation."""
    rets: list[float] = []
    for ep in range(n_episodes):
        ep_ret, _ = run_exit_episode(
            env_id, actor, device, action_space, seed=start_seed + ep, exit_id=exit_id
        )
        rets.append(ep_ret)
    return float(np.mean(rets)) if rets else 0.0


def eval_three_exit_weighted_return(
    env_id: str,
    actor: ActorEENN3Semicircle,
    device: torch.device,
    action_space: spaces.Box,
    *,
    start_seed: int,
    n_episodes: int,
    w1: float,
    w2: float,
    w3: float,
) -> tuple[float, float, float, float]:
    """Return (r_e1, r_e2, r_efull, weighted_return)."""
    r1 = eval_exit_mean_return(
        env_id, actor, device, action_space, start_seed=start_seed, n_episodes=n_episodes, exit_id=1
    )
    r2 = eval_exit_mean_return(
        env_id, actor, device, action_space, start_seed=start_seed, n_episodes=n_episodes, exit_id=2
    )
    r3 = eval_exit_mean_return(
        env_id, actor, device, action_space, start_seed=start_seed, n_episodes=n_episodes, exit_id=3
    )
    weighted = w1 * r1 + w2 * r2 + w3 * r3
    return r1, r2, r3, weighted


def log_train_episodes_all_exits(
    writer: SummaryWriter,
    infos,
    global_step: int,
    *,
    env_id: str,
    actor: ActorEENN3Semicircle,
    device: torch.device,
    action_space: spaces.Box,
    eval_start_seed: int,
    warmup_steps: int,
    episode_counter: list[int],
) -> None:
    """The end of each training episode is recorded as efull; e1/e2 is run after warmup is completed (consistent with joint training/eval)."""
    for info in infos:
        if not (isinstance(info, dict) and "episode" in info):
            continue
        ep = info["episode"]
        writer.add_scalar("train/episode_return_efull", float(ep["r"]), global_step)
        writer.add_scalar("train/episode_len_efull", float(ep["l"]), global_step)
        if global_step < warmup_steps:
            continue
        seed = eval_start_seed + episode_counter[0]
        episode_counter[0] += 1
        r1, l1 = run_exit_episode(
            env_id, actor, device, action_space, seed=seed, exit_id=1
        )
        r2, l2 = run_exit_episode(
            env_id, actor, device, action_space, seed=seed, exit_id=2
        )
        writer.add_scalar("train/episode_return_e1", r1, global_step)
        writer.add_scalar("train/episode_len_e1", l1, global_step)
        writer.add_scalar("train/episode_return_e2", r2, global_step)
        writer.add_scalar("train/episode_len_e2", l2, global_step)


def save_checkpoint(
    path: str,
    *,
    step: int,
    actor: ActorEENN3Semicircle,
    critic: CriticDeepTwin,
    critic_target: CriticDeepTwin,
    opt_a: optim.Optimizer,
    opt_c: optim.Optimizer,
    log_alpha: torch.Tensor,
    env_id: str,
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
        "env_id": env_id,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


_DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semicircle three-port combined SAC (e1/e2/efull)")
    p.add_argument("--env-id", type=str, default="SafetyPointSemicircle0-v5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-timesteps", type=int, default=1_500_000)
    p.add_argument("--warmup-steps", type=int, default=1_000_000, help="Previously actor only efull(exit3) backpass")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--w-exit1", type=float, default=0.2, dest="w_exit1")
    p.add_argument("--w-exit2", type=float, default=0.3, dest="w_exit2")
    p.add_argument("--w-exit3", type=float, default=0.5, dest="w_exit3")
    p.add_argument(
        "--runs-root",
        type=str,
        default=str(_DEFAULT_RUNS_ROOT),
        help="Default DTRL-Off/runs/",
    )
    p.add_argument("--save-freq", type=int, default=150_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-start-seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument(
        "--no-split-joint-grad",
        action="store_true",
        help="The joint stage performs a backward pass on the weighted loss (the default is to split the shared segment gradient)",
    )
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    env_fns = [make_env(args.env_id) for _ in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    assert isinstance(vec_env.action_space, spaces.Box)
    obs_dim = int(np.prod(vec_env.observation_space.shape))
    act_dim = int(np.prod(vec_env.action_space.shape))

    actor = ActorEENN3Semicircle(obs_dim, act_dim).to(device)
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
    run_dir = os.path.join(args.runs_root, f"joint_sac_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    best_model_dir = os.path.join(run_dir, "best_model")
    final_model_dir = os.path.join(run_dir, "final_model")
    for d in (ckpt_dir, tb_dir, best_model_dir, final_model_dir):
        os.makedirs(d, exist_ok=True)

    writer = SummaryWriter(tb_dir)
    best_weighted = -np.inf
    # There is no eval in the warmup phase (e1/e2 has not yet participated in training); the first eval is in global_step >= warmup_steps
    next_eval_at = args.warmup_steps
    next_save_at = args.save_freq

    print(f"[INFO] env_id={args.env_id}")
    print(f"[INFO] total_timesteps={args.total_timesteps}")
    print(f"[INFO] warmup_steps={args.warmup_steps} (efull only)")
    print(f"[INFO] n_envs={args.n_envs}, device={device}")
    print(f"[INFO] save_freq={args.save_freq}, eval_freq={args.eval_freq}")
    print(f"[INFO] eval begins after warmup_steps={args.warmup_steps}")
    print(
        f"[INFO] eval metric (post-warmup): weighted return "
        f"(w1={args.w_exit1}, w2={args.w_exit2}, w3={args.w_exit3})"
    )
    print(f"[INFO] run_dir={run_dir}")

    w1, w2, w3 = args.w_exit1, args.w_exit2, args.w_exit3

    obs = vec_env.reset()
    global_step = 0
    last_save_step = 0
    train_episode_counter = [0]

    pbar = tqdm(
        total=args.total_timesteps,
        unit="step",
        desc="joint_sac",
        dynamic_ncols=True,
        disable=args.no_progress_bar,
    )

    while global_step < args.total_timesteps:
        n = args.n_envs
        if global_step <= args.learning_starts:
            buf_a, env_a = random_scaled_action_batch(vec_env.action_space, n)
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _a1, _lp1, _a2, _lp2, a3, _lp3 = actor.sample_all(o)
                buf_a = a3.cpu().numpy()
            env_a = unscale_action_batch(buf_a, vec_env.action_space)

        next_obs, rewards, dones, infos = vec_env.step(env_a)
        log_train_episodes_all_exits(
            writer,
            infos,
            global_step,
            env_id=args.env_id,
            actor=actor,
            device=device,
            action_space=vec_env.action_space,
            eval_start_seed=args.eval_start_seed,
            warmup_steps=args.warmup_steps,
            episode_counter=train_episode_counter,
        )
        buffer.add_vec(obs, buf_a, rewards, next_obs, dones.astype(np.float32))
        obs = next_obs
        global_step += n
        pbar.update(n)

        do_train = buffer.size >= max(args.batch_size, args.learning_starts) and global_step > args.learning_starts

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
            in_warmup = global_step <= args.warmup_steps
            if in_warmup:
                loss_pi_3.backward()
            elif args.no_split_joint_grad:
                loss_pi.backward()
            else:
                sh, fc2, fc3, deep, e1, e2, e3 = actor.joint_param_groups()
                assign_triple_joint_grads_semicircle(
                    sh, fc2, fc3, deep, e1, e2, e3, loss_pi_1, loss_pi_2, loss_pi_3, w1, w2, w3
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
                args.env_id,
                actor,
                device,
                vec_env.action_space,
                start_seed=args.eval_start_seed,
                n_episodes=args.eval_episodes,
                w1=w1,
                w2=w2,
                w3=w3,
            )
            writer.add_scalar("eval/mean_reward_e1", r1, global_step)
            writer.add_scalar("eval/mean_reward_e2", r2, global_step)
            writer.add_scalar("eval/mean_reward_efull", r3, global_step)
            writer.add_scalar("eval/weighted_mean_return", weighted, global_step)
            print(
                f"[Eval] step={global_step} "
                f"e1={r1:.4f} e2={r2:.4f} efull={r3:.4f} "
                f"weighted({w1:.1f}/{w2:.1f}/{w3:.1f})={weighted:.4f}"
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
                    env_id=args.env_id,
                    extra={
                        "mean_return_e1": r1,
                        "mean_return_e2": r2,
                        "mean_return_efull": r3,
                        "weighted_mean_return": weighted,
                    },
                )
                print(f"[Eval] new best weighted={weighted:.4f} -> {best_path}")

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
                env_id=args.env_id,
                extra={
                    "warmup_steps": args.warmup_steps,
                    "w_exit1": w1,
                    "w_exit2": w2,
                    "w_exit3": w3,
                },
            )
            print(f"[INFO] checkpoint saved: {ckpt_path}")

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
        env_id=args.env_id,
    )
    print(f"[INFO] final_model={final_path}")

    writer.close()
    vec_env.close()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
