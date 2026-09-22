"""Three-port joint SAC on Pendulum-v1: continue the data flow and SB3 alignment settings of pendulum_train.py,
The strategy is ActorEENN3 (exit1/2/3), the value is CriticEENN3 (a pair of Twin-Qs per mouth, a total of 6 Qs),
The default total number of steps is 100k; the first 28k **Actor and Critic are both exit3**: the strategy loss is only propagated back to exit3; Critic TD only accumulates L3 (equivalent weight 0:0:1).
After that, **the same schedule**: by default, it goes through **10k** steps of linear ramp, and Actor/Critic weighting reaches exit1:exit2:exit3 = 2:3:5 (0.2/0.3/0.5).
Data still comes from exit3 to interact with the environment; bootstrap still only uses the exit3 next step strategy and target Q.

The default output directory is: **Same level as this script** `runs/three_exits_joint/<run name>_<timestamp>/` (i.e.
`pendulum/DTRL-Off/Ablation Study/runs/...`); checkpoint every 40k steps;
Starting from 28k, use seed 42–141 to evaluate the weighted selection of three ports every 5k and write it into `best/DTRL-Off_best_model.pt`.

Run: You can add the superior directory to the module path and then execute it (the script has automatically injected ``DTRL-Off`` into ``sys.path``,
No need to copy ``action_utils`` / ``eenn_network`` anymore). Example:
``python "/path/to/Ablation Study/a1_critical.py"``
The log is written to this directory ``Ablation Study/runs/...``"""

from __future__ import annotations

import sys
from pathlib import Path

_DTRL_OFF = Path(__file__).resolve().parent.parent
if str(_DTRL_OFF) not in sys.path:
    sys.path.insert(0, str(_DTRL_OFF))

import argparse
import copy
import os
import random
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from action_utils import scale_action, unscale_action
from eenn_network import ActorEENN3, CriticEENN3, soft_update
from stable_baselines3.common.env_util import make_vec_env


def _z(g: torch.Tensor | None, p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p) if g is None else g


def assign_triple_joint_grads(
    shared: list[torch.nn.Parameter],
    mid: list[torch.nn.Parameter],
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
    """Three ports share the backbone (obs→128) + fc2 (128→128) for exit2/3; trunk_exit3 only exit3; allocate gradients by weight.

Actor topology: exit1 obs-128-out; exit2 obs-128-128-out; exit3 obs-128-128-64-64-out."""
    g1_s = torch.autograd.grad(loss1, shared, retain_graph=True, allow_unused=True)
    g2_s = torch.autograd.grad(loss2, shared, retain_graph=True, allow_unused=True)
    g3_s = torch.autograd.grad(loss3, shared, retain_graph=True, allow_unused=True)
    g1_e1 = torch.autograd.grad(loss1, e1, retain_graph=True, allow_unused=True)
    g2_m = torch.autograd.grad(loss2, mid, retain_graph=True, allow_unused=True)
    g2_e2 = torch.autograd.grad(loss2, e2, retain_graph=True, allow_unused=True)
    g3_m = torch.autograd.grad(loss3, mid, retain_graph=True, allow_unused=True)
    g3_d = torch.autograd.grad(loss3, deep, retain_graph=True, allow_unused=True)
    g3_e3 = torch.autograd.grad(loss3, e3, retain_graph=False, allow_unused=True)

    for p, a, b, c in zip(shared, g1_s, g2_s, g3_s):
        p.grad = w1 * _z(a, p) + w2 * _z(b, p) + w3 * _z(c, p)
    for p, b, c in zip(mid, g2_m, g3_m):
        p.grad = w2 * _z(b, p) + w3 * _z(c, p)
    for p, g in zip(deep, g3_d):
        p.grad = _z(g, p)
    for p, g in zip(e1, g1_e1):
        p.grad = _z(g, p)
    for p, g in zip(e2, g2_e2):
        p.grad = _z(g, p)
    for p, g in zip(e3, g3_e3):
        p.grad = _z(g, p)


def scheduled_triple_actor_weights(
    step: int,
    warmup_steps: int,
    schedule_steps: int,
    w1_final: float,
    w2_final: float,
    w3_final: float,
) -> tuple[float, float, float, str]:
    """Only exit3 in warmup; linear transition from (0,0,1) to (w1f,w2f,w3f) in ramp; steady is fixed."""
    if step <= warmup_steps:
        return 0.0, 0.0, 1.0, "warmup_exit3_only"
    if schedule_steps <= 0:
        return w1_final, w2_final, w3_final, "steady"
    end = warmup_steps + schedule_steps
    if step > end:
        return w1_final, w2_final, w3_final, "steady"
    t = (step - warmup_steps) / float(schedule_steps)
    t = min(1.0, max(0.0, t))
    w1 = t * w1_final
    w2 = t * w2_final
    w3 = (1.0 - t) * 1.0 + t * w3_final
    return w1, w2, w3, "ramp"


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


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
        self.dones[i] = float(t.done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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


def eval_exit_return(
    env_id: str,
    actor: ActorEENN3,
    device: torch.device,
    seed: int,
    episodes: int,
    exit_id: int,
) -> float:
    """Use the specified exit (1/2/3) deterministic mean action to do a short evaluation and return the average undiscounted episode return."""
    assert exit_id in (1, 2, 3)
    env = gym.make(env_id)
    assert isinstance(env.action_space, spaces.Box)
    rets: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                m1, _ls1, m2, _ls2, m3, _ls3 = actor.forward(o)
                if exit_id == 1:
                    m = m1
                elif exit_id == 2:
                    m = m2
                else:
                    m = m3
                a_scaled = torch.tanh(m).cpu().numpy().flatten()
                action_env = unscale_action(a_scaled, env.action_space)
            obs, reward, terminated, truncated, _ = env.step(action_env)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
        rets.append(ep_ret)
    env.close()
    return float(np.mean(rets)) if rets else 0.0


def eval_three_exit_mean_returns(
    env_id: str,
    actor: ActorEENN3,
    device: torch.device,
    *,
    start_seed: int,
    n_seeds: int,
) -> tuple[float, float, float]:
    """seed starts from start_seed for n_seeds consecutive episodes, and obtains the average undiscounted return of exit1/2/3 respectively."""
    r1 = eval_exit_return(env_id, actor, device, start_seed, n_seeds, 1)
    r2 = eval_exit_return(env_id, actor, device, start_seed, n_seeds, 2)
    r3 = eval_exit_return(env_id, actor, device, start_seed, n_seeds, 3)
    return r1, r2, r3


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.total_steps % args.n_envs != 0:
        raise SystemExit(
            f"--total-steps ({args.total_steps}) must be accessible to --n-envs ({args.n_envs}) divisible"
        )

    vec_env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
    obs_dim = int(np.prod(vec_env.observation_space.shape))
    act_dim = int(np.prod(vec_env.action_space.shape))

    actor = ActorEENN3(obs_dim, act_dim).to(device)
    critic = CriticEENN3(obs_dim, act_dim).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad = False

    opt_a = optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = optim.Adam(critic.parameters(), lr=args.lr)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -float(act_dim)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    # Create a separate directory for each run: {log_root}/three_exits_joint/<name>_<timestamp>/ to write to TensorBoard, checkpoints/ to save weights
    base_name = args.run_name or f"eenn_joint_{args.env_id.replace('/', '_')}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.log_root, "three_exits_joint")
    log_dir = os.path.join(run_root, f"{base_name}_{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    best_dir = os.path.join(log_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[three_exits_joint] TensorBoard: {log_dir}")
    print(f"[three_exits_joint] Checkpoints: {ckpt_dir}")
    print(f"[three_exits_joint] Best (weighted eval): {best_dir}")
    best_weighted = -np.inf

    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    n_envs = args.n_envs
    ep_ret = np.zeros(n_envs, dtype=np.float64)
    ep_len = np.zeros(n_envs, dtype=np.int32)
    last_ep_ret: float | None = None
    env_step = 0
    last_log_at = 0

    pbar = tqdm(
        total=args.total_steps,
        unit="env_step",
        desc="three_exits_joint",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    while env_step < args.total_steps:
        assert isinstance(vec_env.action_space, spaces.Box)

        if env_step < args.learning_starts:
            actions_env = np.stack(
                [vec_env.action_space.sample().astype(np.float32) for _ in range(n_envs)]
            )
            buf_actions = np.stack(
                [scale_action(actions_env[i], vec_env.action_space) for i in range(n_envs)]
            )
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _a1, _lp1, _a2, _lp2, a3, _lp3 = actor.sample_all(o)
                buf_actions = a3.cpu().numpy()
            actions_env = np.stack(
                [
                    unscale_action(buf_actions[i], vec_env.action_space)
                    for i in range(n_envs)
                ]
            )

        step_out = vec_env.step(actions_env)
        if len(step_out) == 5:
            next_obs, rewards, terminated, truncated, _infos = step_out
            dones = np.logical_or(terminated, truncated)
        else:
            next_obs, rewards, dones, _infos = step_out

        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)

        step_tag = env_step + n_envs

        for i in range(n_envs):
            buffer.add(
                Transition(
                    obs=np.asarray(obs[i], dtype=np.float32).reshape(-1),
                    action=np.asarray(buf_actions[i], dtype=np.float32).reshape(-1),
                    reward=float(rewards[i]),
                    next_obs=np.asarray(next_obs[i], dtype=np.float32).reshape(-1),
                    done=bool(dones[i]),
                )
            )
            ep_ret[i] += rewards[i]
            ep_len[i] += 1
            if dones[i]:
                writer.add_scalar("train/episode_return", float(ep_ret[i]), step_tag)
                writer.add_scalar("train/episode_len", int(ep_len[i]), step_tag)
                last_ep_ret = float(ep_ret[i])
                ep_ret[i] = 0.0
                ep_len[i] = 0

        obs = next_obs
        env_step += n_envs
        pbar.update(n_envs)

        if buffer.size < args.batch_size or env_step <= args.learning_starts:
            continue

        alpha = log_alpha.exp().clamp(min=1e-8)
        b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(args.batch_size, device)

        # Share the same set of ramps with Actor: warmup (0,0,1), steady (w_exit1,w_exit2,w_exit3)
        w1, w2, w3, phase = scheduled_triple_actor_weights(
            env_step,
            args.warmup_steps,
            args.schedule_steps,
            args.w_exit1,
            args.w_exit2,
            args.w_exit3,
        )

        with torch.no_grad():
            _a1n, _lp1n, _a2n, _lp2n, a3_next, log_pi3_next = actor.sample_all(b_next_obs)
            _p1t, _p2t, p3_t = critic_target.triple(b_next_obs, a3_next)
            q1_3t, q2_3t = p3_t
            q_min_3 = torch.min(q1_3t, q2_3t)
            target_q = b_rew + (1.0 - b_done) * args.gamma * (q_min_3 - alpha * log_pi3_next)

        # Critic: L1/L2/L3 independent TD; weighting coefficient is consistent with Actor (warmup only has gradient for L3)
        (q1_1, q2_1), (q1_2, q2_2), (q1_3, q2_3) = critic.triple(b_obs, b_act)
        loss_c1 = 0.5 * (F.mse_loss(q1_1, target_q) + F.mse_loss(q2_1, target_q))
        loss_c2 = 0.5 * (F.mse_loss(q1_2, target_q) + F.mse_loss(q2_2, target_q))
        loss_c3 = 0.5 * (F.mse_loss(q1_3, target_q) + F.mse_loss(q2_3, target_q))
        loss_c = w1 * loss_c1 + w2 * loss_c2 + w3 * loss_c3

        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

        a1, log_pi1, a2, log_pi2, a3, log_pi3 = actor.sample_all(b_obs)
        (q1_1a, q2_1a), _, _ = critic.triple(b_obs, a1)
        _, (q1_2a, q2_2a), _ = critic.triple(b_obs, a2)
        _, _, (q1_3a, q2_3a) = critic.triple(b_obs, a3)
        qm1 = torch.min(q1_1a, q2_1a)
        qm2 = torch.min(q1_2a, q2_2a)
        qm3 = torch.min(q1_3a, q2_3a)

        loss_pi_1 = (alpha * log_pi1 - qm1).mean()
        loss_pi_2 = (alpha * log_pi2 - qm2).mean()
        loss_pi_3 = (alpha * log_pi3 - qm3).mean()

        loss_pi = w1 * loss_pi_1 + w2 * loss_pi_2 + w3 * loss_pi_3

        opt_a.zero_grad()
        if phase == "warmup_exit3_only":
            loss_pi_3.backward()
        elif args.split_joint_grad:
            sh, mid, deep, e1, e2, e3 = actor.joint_param_groups()
            assign_triple_joint_grads(
                sh,
                mid,
                deep,
                e1,
                e2,
                e3,
                loss_pi_1,
                loss_pi_2,
                loss_pi_3,
                w1,
                w2,
                w3,
            )
        else:
            loss_pi.backward()
        opt_a.step()

        loss_alpha = -(log_alpha * (log_pi3.detach() + target_entropy)).mean()
        opt_alpha.zero_grad()
        loss_alpha.backward()
        opt_alpha.step()

        soft_update(critic_target, critic, args.tau)

        # Every eval_freq "environment cumulative step" starting from eval_begin_step: three-port evaluation + weighted best
        if (
            env_step >= args.eval_begin_step
            and (env_step - args.eval_begin_step) % args.eval_freq == 0
        ):
            r1, r2, r3 = eval_three_exit_mean_returns(
                args.env_id,
                actor,
                device,
                start_seed=args.eval_start_seed,
                n_seeds=args.n_eval_seeds,
            )
            w_score = (
                args.w_exit1 * r1 + args.w_exit2 * r2 + args.w_exit3 * r3
            )
            writer.add_scalar("reward_eval/exit1_mean_return", r1, env_step)
            writer.add_scalar("reward_eval/exit2_mean_return", r2, env_step)
            writer.add_scalar("reward_eval/exit3_mean_return", r3, env_step)
            writer.add_scalar("reward_eval/weighted_mean_return", w_score, env_step)
            if w_score > best_weighted:
                best_weighted = w_score
                best_path = os.path.join(best_dir, "best_model.pt")
                torch.save(
                    {
                        "env_step": env_step,
                        "weighted_score": w_score,
                        "mean_return_exit1": r1,
                        "mean_return_exit2": r2,
                        "mean_return_exit3": r3,
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "critic_target": critic_target.state_dict(),
                        "opt_a": opt_a.state_dict(),
                        "opt_c": opt_c.state_dict(),
                        "log_alpha": log_alpha.detach(),
                        "env_id": args.env_id,
                        "n_envs": args.n_envs,
                    },
                    best_path,
                )
                print(
                    f"[three_exits_joint] new best weighted_return={w_score:.4f} "
                    f"(e1={r1:.2f} e2={r2:.2f} e3={r3:.2f}) -> {best_path}"
                )

        if env_step - last_log_at >= args.log_interval:
            last_log_at = env_step
            writer.add_scalar("loss/critic", loss_c.item(), env_step)
            writer.add_scalar("loss/actor", loss_pi.item(), env_step)
            writer.add_scalar("loss/actor_exit3", loss_pi_3.item(), env_step)
            writer.add_scalar("loss/alpha", loss_alpha.item(), env_step)
            writer.add_scalar("alpha/value", alpha.item(), env_step)
            writer.add_scalar("loss/critic_exit3", loss_c3.item(), env_step)
            writer.add_scalar("schedule/w_exit1", w1, env_step)
            writer.add_scalar("schedule/w_exit2", w2, env_step)
            writer.add_scalar("schedule/w_exit3", w3, env_step)
            ph = {"warmup_exit3_only": 0.0, "ramp": 1.0, "steady": 2.0}[phase]
            writer.add_scalar("schedule/phase", ph, env_step)
            if not args.no_progress:
                ep_str = f"{last_ep_ret:.1f}" if last_ep_ret is not None else "-"
                pbar.set_postfix_str(
                    f"{phase} w1={w1:.2f} w2={w2:.2f} w3={w3:.2f} ep={ep_str} Lc={loss_c.item():.3f} Lpi={loss_pi.item():.3f}",
                    refresh=True,
                )

        if env_step % args.save_interval == 0 and env_step > 0:
            ckpt = {
                "env_step": env_step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
                "opt_a": opt_a.state_dict(),
                "opt_c": opt_c.state_dict(),
                "log_alpha": log_alpha.detach(),
                "env_id": args.env_id,
                "n_envs": args.n_envs,
                "w_exit1": args.w_exit1,
                "w_exit2": args.w_exit2,
                "w_exit3": args.w_exit3,
                "warmup_steps": args.warmup_steps,
                "schedule_steps": args.schedule_steps,
                "split_joint_grad": args.split_joint_grad,
            }
            path = os.path.join(ckpt_dir, f"ckpt_{env_step}.pt")
            torch.save(ckpt, path)
            writer.add_text("checkpoint", path, env_step)

    pbar.close()
    writer.close()
    vec_env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument(
        "--n-envs",
        type=int,
        default=4,
        dest="n_envs",
        help="Number of parallel environments (SB3 VecEnv); total-steps must be an integer multiple of it",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=28_000,
        help="Previously, actor only had exit3 loss backpropagation; later, it was connected with --schedule-steps ramp.",
    )
    p.add_argument(
        "--schedule-steps",
        type=int,
        default=10_000,
        help="Linear ramp: Actor/Critic weighting transitions from (0,0,1) to (--w-exit1/2/3), default 10k steps",
    )
    p.add_argument(
        "--w-exit1",
        type=float,
        default=0.2,
        dest="w_exit1",
        help="steady target weight; default 0.2 in 2:3:5",
    )
    p.add_argument(
        "--w-exit2",
        type=float,
        default=0.3,
        dest="w_exit2",
        help="steady target weight; default 0.3 in 2:3:5",
    )
    p.add_argument(
        "--w-exit3",
        type=float,
        default=0.5,
        dest="w_exit3",
        help="steady target weight; default 0.5 in 2:3:5",
    )
    p.add_argument("--log-interval", type=int, default=500)
    # Similar to this script: Ablation Study/runs/
    _default_runs = Path(__file__).resolve().parent / "runs"
    p.add_argument(
        "--log-root",
        type=str,
        default=str(_default_runs),
        help="Log root directory; default DTRL-Off/Ablation Study/runs",
    )
    p.add_argument(
        "--eval-begin-step",
        type=int,
        default=28_000,
        help="From this step, press eval-freq to do three seed evaluations",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluation interval (environment step)",
    )
    p.add_argument(
        "--eval-start-seed",
        type=int,
        default=42,
        help="Evaluate episode starting from seed, consecutive n-eval-seeds",
    )
    p.add_argument(
        "--n-eval-seeds",
        type=int,
        default=100,
        dest="n_eval_seeds",
        help="The default is 100, which is seed 42..141",
    )
    p.add_argument("--save-interval", type=int, default=40_000)
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run name (without timestamp); if empty, use eenn_joint_<env-id>",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--no-split-joint-grad",
        action="store_true",
        help="ramp/steady: perform a backward step on the weighted loss; the default is to separate the shared backbone gradient",
    )
    args = p.parse_args()
    args.split_joint_grad = not args.no_split_joint_grad
    return args


if __name__ == "__main__":
    train(parse_args())
