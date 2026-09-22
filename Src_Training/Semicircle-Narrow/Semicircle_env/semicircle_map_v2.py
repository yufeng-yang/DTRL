"""Semi-circular track v2: Same geometry as v0, but more densely sampled arc hazards."""

import numpy as np

# ---------- Semicircular ring parameters ----------
RADIUS = 2.0   # Centerline radius (lane center)
GOAL_SIZE = 0.25
GOAL_X = RADIUS
GOAL_Y = 0.0

# The starting point is on the arc (slightly away from the opening), because the opening has been completely blocked by the retaining wall
START_ANGLE = np.pi - 0.12
START_X = float(RADIUS * np.cos(START_ANGLE))
START_Y = float(RADIUS * np.sin(START_ANGLE))

LANE_HALF_WIDTH = 0.35   # Same as v0

# ---------- Small section of boundary wall ----------
SEGMENT_SPACING = 0.10
SEGMENT_HALF_X = 0.06
SEGMENT_HALF_Y = 0.04
SEGMENT_HALF_Z = 1e-2


def _start_gate_segment_samples():
    """The starting point opening is completely blocked with hazard, leaving no opening."""
    offset_out = LANE_HALF_WIDTH + SEGMENT_HALF_Y
    r_inner = RADIUS - offset_out
    r_outer = RADIUS + offset_out
    gate_y = -SEGMENT_HALF_Y - 0.12
    x_lo = -r_outer + SEGMENT_HALF_X
    x_hi = -r_inner - SEGMENT_HALF_X
    n = max(2, int(round((x_hi - x_lo) / (SEGMENT_SPACING * 0.8))) + 1)
    locs = [(float(x), gate_y) for x in np.linspace(x_lo, x_hi, n)]
    rots = [0.0] * len(locs)
    return locs, rots


def _centerline_samples():
    """Center line of the upper semicircle: According to the spacing setting, the sampling is really spread out and the gaps between blocks are reduced."""
    n = max(12, int(np.ceil(np.pi * RADIUS / SEGMENT_SPACING)))
    thetas = np.linspace(np.pi, 0, n + 1)
    points, tangents = [], []
    for theta in thetas:
        x = RADIUS * np.cos(theta)
        y = RADIUS * np.sin(theta)
        th = np.arctan2(-np.cos(theta), np.sin(theta))
        points.append((float(x), float(y)))
        tangents.append(float(th))
    return points, tangents


def segment_center_locations():
    """Semi-circular ring: only the upper half-circle (inner arc + outer arc), the lower side is a diameter opening without sealing."""
    centers, tangents = _centerline_samples()
    offset_out = LANE_HALF_WIDTH + SEGMENT_HALF_Y
    left_locs, right_locs = [], []
    left_rots, right_rots = [], []
    for (x, y), _ in zip(centers, tangents):
        theta = np.arctan2(y, x) if (x != 0 or y != 0) else 0.0
        nx_in = -np.cos(theta)
        ny_in = -np.sin(theta)
        lx = x + offset_out * nx_in
        ly = y + offset_out * ny_in
        rx = x - offset_out * nx_in
        ry = y - offset_out * ny_in
        left_locs.append((float(lx), float(ly)))
        right_locs.append((float(rx), float(ry)))
        rot = theta + np.pi / 2
        left_rots.append(float(rot))
        right_rots.append(float(rot))
    gate_locs, gate_rots = _start_gate_segment_samples()
    locations = left_locs + right_locs + gate_locs
    rotations = left_rots + right_rots + gate_rots
    return locations, rotations


def start_xy():
    return (START_X, START_Y)


def start_rot():
    return float(np.arctan2(-np.cos(START_ANGLE), np.sin(START_ANGLE)))


def goal_xy():
    return (GOAL_X, GOAL_Y)
