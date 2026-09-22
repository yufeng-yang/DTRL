"""Go2 Box action: replay saves [-1,1], unscale to clip scale when interacting with the environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxSpace:
    """Lightweight Box action space (does not rely on gymnasium)."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def symmetric(cls, clip: float, dim: int) -> BoxSpace:
        c = float(clip)
        return cls(
            low=np.full(dim, -c, dtype=np.float32),
            high=np.full(dim, c, dtype=np.float32),
        )

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)


def scale_action(action: np.ndarray, action_space: BoxSpace) -> np.ndarray:
    low, high = action_space.low.astype(np.float64), action_space.high.astype(np.float64)
    return (2.0 * ((action.astype(np.float64) - low) / (high - low)) - 1.0).astype(np.float32)


def unscale_action(scaled_action: np.ndarray, action_space: BoxSpace) -> np.ndarray:
    low, high = action_space.low.astype(np.float64), action_space.high.astype(np.float64)
    return (low + (0.5 * (scaled_action.astype(np.float64) + 1.0) * (high - low))).astype(
        np.float32
    )


def unscale_action_batch(scaled: np.ndarray, action_space: BoxSpace) -> np.ndarray:
    low = action_space.low.astype(np.float64)
    high = action_space.high.astype(np.float64)
    return (low + 0.5 * (scaled.astype(np.float64) + 1.0) * (high - low)).astype(np.float32)


def random_scaled_action_batch(
    action_space: BoxSpace, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    unscaled = np.stack(
        [action_space.sample().astype(np.float32).reshape(-1) for _ in range(batch_size)],
        axis=0,
    )
    scaled = np.stack([scale_action(unscaled[i], action_space) for i in range(batch_size)], axis=0)
    return scaled, unscaled
