# Semicircle_env: Semicircle track

Standard semi-circular arc track: center (0,0), radius `RADIUS`, arc from starting point (-R,0) to end point (R,0).

## use

```python
import Semicircle_env
import safety_gymnasium

env = safety_gymnasium.make('SafetyCarSemicircle0-v1', render_mode='human')
```

Environment ID: `SafetyPointSemicircle0-v1`, `SafetyCarSemicircle0-v1`, `SafetyRacecarSemicircle0-v1`.

## document

| Documentation | Description |
|------|------|
| semicircle_map.py | Semicircle parameters (radius, start/end point, lane half-width), centerline sampling, segment_center_locations() |
| semicircle_assets.py | PathSegmentHazards along arc boundary walls |
| semicircle_task.py | SemicircleLevel0 tasks and rewards |
| __init__.py | Register tasks and environment |

## Parameters (semicircle_map.py)

- `RADIUS`: semicircle radius, default 1.0
- `LANE_HALF_WIDTH`: Lane half width
- `START_X/Y`, `GOAL_X/Y`: starting point (-R,0), end point (R,0)
