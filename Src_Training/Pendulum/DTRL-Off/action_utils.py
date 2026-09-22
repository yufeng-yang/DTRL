"""Consistent with Stable Baselines3: Box actions store [-1,1] in replay and use physical scales when interacting with the environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def buffer_and_env_action_box(
    env,
    step: int,
    learning_starts: int,
    scaled_policy_action: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned with SB3 _sample_action:
    - step <= learning_starts: The environment action space is uniformly random, and replay saves scale [-1,1];
    - Otherwise: the policy output is [-1,1], replay stores this value, and the environment uses unscale.

    Return (buffer_action_scaled, env_action_unscaled).    """
    assert isinstance(env.action_space, spaces.Box)
    if step <= learning_starts:
        unscaled = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
        buf = scale_action(unscaled, env.action_space)
        return buf, unscaled
    assert scaled_policy_action is not None
    sa = np.asarray(scaled_policy_action, dtype=np.float32).reshape(-1)
    unscaled = unscale_action(sa, env.action_space)
    return sa, unscaled


def scale_action(action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    """[low, high] → [-1, 1]"""
    low, high = action_space.low.astype(np.float64), action_space.high.astype(np.float64)
    return (2.0 * ((action.astype(np.float64) - low) / (high - low)) - 1.0).astype(np.float32)


def unscale_action(scaled_action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    """[-1, 1] → [low, high]"""
    low, high = action_space.low.astype(np.float64), action_space.high.astype(np.float64)
    return (
        low + (0.5 * (scaled_action.astype(np.float64) + 1.0) * (high - low))
    ).astype(np.float32)
