"""Three-port joint SAC on Pendulum-v1: continue the data flow and SB3 alignment settings of pendulum_train.py,
The strategy is ActorEENN3Deep (exit1/2/3: obs-128-128-out/obs-128-128-128-out/obs-128-128-128-64-64-out),
Value is single Twin-Q (CriticDeepTwin, deep aligned with exit3).
Actor **only exit3** backpropagation within the first ``--warmup-steps`` (default 28k); thereafter use weighted joint loss (--w-exit1/2/3).

Default log root directory: pendulum/DTRL-Off/runs/three_exits_joint/<run name>_<timestamp>/;
Checkpoint every 40k steps; ``reward_eval/*`` writes three **1 round** rewards at the end of each training episode (synchronized with ``train/episode_return`` to change the density),
Also write **multi-seed averaging** in ``--eval-freq`` (consistent with the best model). You can use ``--no-reward-eval-per-episode`` to turn off per-episode evaluation to speed up.

Run: conda activate sb3sg && python joint_train.py"""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
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

from action_utils import buffer_and_env_action_box, unscale_action
from eenn_network import ActorEENN3Deep, CriticDeepTwin, soft_update


def _z(g: torch.Tensor | None, p: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(p) if g is None else g


def assign_triple_joint_grads_deep(
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
    """The shared segment gradient of ActorEENN3Deep is combined with weight according to loss1/2/3; fc3 only loses 2/3, deep only loses 3."""
    g1_e1 = torch.autograd.grad(loss1, e1, retain_graph=True, allow_unused=True)
    g1_f2 = torch.autograd.grad(loss1, fc2, retain_graph=True, allow_unused=True)
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
    for p, a, b, c in zip(fc2, g1_f2, g2_f2, g3_f2):
        p.grad = w1 * _z(a, p) + w2 * _z(b, p) + w3 * _z(c, p)
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
    actor: ActorEENN3Deep,
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
    actor: ActorEENN3Deep,
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

    env = gym.make(args.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    actor = ActorEENN3Deep(obs_dim, act_dim).to(device)
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

    r1_0, r2_0, r3_0 = eval_three_exit_mean_returns(
        args.env_id,
        actor,
        device,
        start_seed=args.eval_start_seed,
        n_seeds=args.n_eval_seeds,
    )
    w0 = args.w_exit1 * r1_0 + args.w_exit2 * r2_0 + args.w_exit3 * r3_0
    writer.add_scalar("reward_eval/exit1_mean_return", r1_0, 0)
    writer.add_scalar("reward_eval/exit2_mean_return", r2_0, 0)
    writer.add_scalar("reward_eval/exit3_mean_return", r3_0, 0)
    writer.add_scalar("reward_eval/weighted_mean_return", w0, 0)

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    last_ep_ret: float | None = None

    step_iter = tqdm(
        range(1, args.total_steps + 1),
        total=args.total_steps,
        unit="step",
        desc="three_exits_joint",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for step in step_iter:
        if isinstance(env.action_space, spaces.Box):
            if step <= args.learning_starts:
                buf_a, action_env = buffer_and_env_action_box(
                    env, step, args.learning_starts, None
                )
            else:
                with torch.no_grad():
                    o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _a1, _lp1, _a2, _lp2, a3, _lp3 = actor.sample_all(o)
                    a_np = a3.cpu().numpy().flatten()
                buf_a, action_env = buffer_and_env_action_box(
                    env, step, args.learning_starts, a_np
                )
        else:
            if step <= args.learning_starts:
                action_env = env.action_space.sample()
                buf_a = np.asarray(action_env, dtype=np.float32).reshape(-1)
            else:
                with torch.no_grad():
                    o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _a1, _lp1, _a2, _lp2, a3, _lp3 = actor.sample_all(o)
                    buf_a = a3.cpu().numpy().flatten()
                action_env = buf_a

        next_obs, reward, terminated, truncated, _ = env.step(action_env)
        done = bool(terminated or truncated)
        buffer.add(
            Transition(
                obs=np.asarray(obs, dtype=np.float32).reshape(-1),
                action=np.asarray(buf_a, dtype=np.float32).reshape(-1),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).reshape(-1),
                done=done,
            )
        )
        ep_return += float(reward)
        ep_len += 1
        obs = next_obs
        if done:
            writer.add_scalar("train/episode_return", ep_return, step)
            writer.add_scalar("train/episode_len", ep_len, step)
            last_ep_ret = float(ep_return)
            if args.reward_eval_per_episode:
                roll_seed = args.eval_start_seed + step
                r1e = eval_exit_return(
                    args.env_id, actor, device, roll_seed, 1, 1
                )
                r2e = eval_exit_return(
                    args.env_id, actor, device, roll_seed, 1, 2
                )
                r3e = eval_exit_return(
                    args.env_id, actor, device, roll_seed, 1, 3
                )
                we = (
                    args.w_exit1 * r1e
                    + args.w_exit2 * r2e
                    + args.w_exit3 * r3e
                )
                writer.add_scalar("reward_eval/exit1_mean_return", r1e, step)
                writer.add_scalar("reward_eval/exit2_mean_return", r2e, step)
                writer.add_scalar("reward_eval/exit3_mean_return", r3e, step)
                writer.add_scalar("reward_eval/weighted_mean_return", we, step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

        if buffer.size < args.batch_size or step <= args.learning_starts:
            continue

        alpha = log_alpha.exp().clamp(min=1e-8)
        b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(args.batch_size, device)

        with torch.no_grad():
            _a1n, _lp1n, _a2n, _lp2n, a3_next, log_pi3_next = actor.sample_all(b_next_obs)
            q1_t, q2_t = critic_target(b_next_obs, a3_next)
            q_min_t = torch.min(q1_t, q2_t)
            target_q = b_rew + (1.0 - b_done) * args.gamma * (
                q_min_t - alpha * log_pi3_next
            )

        q1, q2 = critic(b_obs, b_act)
        loss_c = 0.5 * (
            F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        )

        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

        a1, log_pi1, a2, log_pi2, a3, log_pi3 = actor.sample_all(b_obs)
        q1_1a, q2_1a = critic(b_obs, a1)
        q1_2a, q2_2a = critic(b_obs, a2)
        q1_3a, q2_3a = critic(b_obs, a3)
        qm1 = torch.min(q1_1a, q2_1a)
        qm2 = torch.min(q1_2a, q2_2a)
        qm3 = torch.min(q1_3a, q2_3a)

        loss_pi_1 = (alpha * log_pi1 - qm1).mean()
        loss_pi_2 = (alpha * log_pi2 - qm2).mean()
        loss_pi_3 = (alpha * log_pi3 - qm3).mean()

        w1, w2, w3 = args.w_exit1, args.w_exit2, args.w_exit3
        loss_pi = w1 * loss_pi_1 + w2 * loss_pi_2 + w3 * loss_pi_3

        opt_a.zero_grad()
        if step <= args.warmup_steps:
            loss_pi_3.backward()
        elif args.split_joint_grad:
            sh, fc2, fc3, deep, e1, e2, e3 = actor.joint_param_groups()
            assign_triple_joint_grads_deep(
                sh,
                fc2,
                fc3,
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

        if (
            step >= args.eval_begin_step
            and (step - args.eval_begin_step) % args.eval_freq == 0
        ):
            r1, r2, r3 = eval_three_exit_mean_returns(
                args.env_id,
                actor,
                device,
                start_seed=args.eval_start_seed,
                n_seeds=args.n_eval_seeds,
            )
            w_score = args.w_exit1 * r1 + args.w_exit2 * r2 + args.w_exit3 * r3
            writer.add_scalar("reward_eval/exit1_mean_return", r1, step)
            writer.add_scalar("reward_eval/exit2_mean_return", r2, step)
            writer.add_scalar("reward_eval/exit3_mean_return", r3, step)
            writer.add_scalar("reward_eval/weighted_mean_return", w_score, step)
            if w_score > best_weighted:
                best_weighted = w_score
                best_path = os.path.join(best_dir, "best_model.pt")
                torch.save(
                    {
                        "step": step,
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
                    },
                    best_path,
                )
                print(
                    f"[three_exits_joint] new best weighted_return={w_score:.4f} "
                    f"(e1={r1:.2f} e2={r2:.2f} e3={r3:.2f}) -> {best_path}"
                )

        if step % args.log_interval == 0:
            in_warmup = step <= args.warmup_steps
            writer.add_scalar("loss/critic", loss_c.item(), step)
            writer.add_scalar(
                "loss/actor",
                loss_pi_3.item() if in_warmup else loss_pi.item(),
                step,
            )
            writer.add_scalar("loss/actor_exit1", loss_pi_1.item(), step)
            writer.add_scalar("loss/actor_exit2", loss_pi_2.item(), step)
            writer.add_scalar("loss/actor_exit3", loss_pi_3.item(), step)
            writer.add_scalar("loss/alpha", loss_alpha.item(), step)
            writer.add_scalar("alpha/value", alpha.item(), step)
            writer.add_scalar("warmup/active", 1.0 if in_warmup else 0.0, step)
            writer.add_scalar("weights/w_exit1", w1, step)
            writer.add_scalar("weights/w_exit2", w2, step)
            writer.add_scalar("weights/w_exit3", w3, step)
            if not args.no_progress:
                ep_str = f"{last_ep_ret:.1f}" if last_ep_ret is not None else "-"
                phase = "warmup_e3" if in_warmup else "joint"
                lip = loss_pi_3.item() if in_warmup else loss_pi.item()
                step_iter.set_postfix_str(
                    f"{phase} w=({w1:.2f},{w2:.2f},{w3:.2f}) ep={ep_str} Lc={loss_c.item():.3f} Lpi={lip:.3f}",
                    refresh=True,
                )

        if step % args.save_interval == 0:
            ckpt = {
                "step": step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
                "opt_a": opt_a.state_dict(),
                "opt_c": opt_c.state_dict(),
                "log_alpha": log_alpha.detach(),
                "env_id": args.env_id,
                "w_exit1": args.w_exit1,
                "w_exit2": args.w_exit2,
                "w_exit3": args.w_exit3,
                "warmup_steps": args.warmup_steps,
                "split_joint_grad": args.split_joint_grad,
            }
            path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
            torch.save(ckpt, path)
            writer.add_text("checkpoint", path, step)

    writer.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-steps", type=int, default=100_000)
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
        help="Previously, actor only used exit3 loss backpropagation; later, --w-exit1/2/3 weighted joint loss was used.",
    )
    p.add_argument(
        "--w-exit1",
        type=float,
        default=0.2,
        dest="w_exit1",
        help="exit1 weight in actor joint loss; default 0.2 in 2:3:5",
    )
    p.add_argument(
        "--w-exit2",
        type=float,
        default=0.3,
        dest="w_exit2",
        help="exit2 weight in actor joint loss",
    )
    p.add_argument(
        "--w-exit3",
        type=float,
        default=0.5,
        dest="w_exit3",
        help="exit3 weights in actor joint loss",
    )
    p.add_argument("--log-interval", type=int, default=500)
    _default_runs = Path(__file__).resolve().parent / "runs"
    p.add_argument(
        "--log-root",
        type=str,
        default=str(_default_runs),
        help="Log root directory; default pendulum/DTRL-Off/runs",
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
    p.add_argument("--save-interval", type=int, default=25_000)
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Run name (without timestamp); if empty, use eenn_joint_<env-id>",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--no-reward-eval-per-episode",
        action="store_true",
        help="Do not run three single-game evaluations at the end of each training episode (reward_eval only step0 + timing multi-seed)",
    )
    p.add_argument(
        "--no-split-joint-grad",
        action="store_true",
        help="One backward for weighted loss_pi; defaults to splitting gradients by shared segments",
    )
    args = p.parse_args()
    args.split_joint_grad = not args.no_split_joint_grad
    args.reward_eval_per_episode = not args.no_reward_eval_per_episode
    return args


if __name__ == "__main__":
    train(parse_args())
