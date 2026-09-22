# Semicircle track task: along the upper semicircle arc from (-R,0) to (R,0), hitting the wall will cost you

from copy import deepcopy

import mujoco
import numpy as np

from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.world import World

from .semicircle_map import GOAL_SIZE, goal_xy, start_xy, start_rot, segment_center_locations
from .semicircle_assets import PathSegmentHazards


class SemicircleLevel0(BaseTask):
    """A standard upper semicircle track, starting point (-R,0), end point (R,0), hitting the boundary is recorded as cost_hazards."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        from .semicircle_map import RADIUS, LANE_HALF_WIDTH

        self.agent.locations = [start_xy()]
        self.agent.rot = start_rot()
        self.agent.keepout = 0.0
        margin = RADIUS + LANE_HALF_WIDTH + 0.2
        self.placements_conf.extents = [-margin, -0.2, margin, margin]
        self.placements_conf.margin = 0.0

        gx, gy = goal_xy()
        self._add_geoms(Goal(size=GOAL_SIZE, keepout=GOAL_SIZE * 0.5, locations=[(gx, gy)]))

        seg_locs, self._hazard_rots = segment_center_locations()
        self._add_geoms(
            PathSegmentHazards(num=len(seg_locs), locations=seg_locs, keepout=0.0)
        )
        self._seg_locs = seg_locs

        self.last_dist_goal = None
        self.mechanism_conf.continue_goal = False
        self.reward_conf.reward_clip = None
        self._reward_goal = 10.0
        self._reward_hit = -1.0
        self._reward_step = -0.01
        self._reward_distance_scale = 0.5

    def _build_placements_dict(self) -> None:
        """Only the agent and goal are randomly placed, and the hazard position is fixed and injected into the layout by _build to avoid ResamplingError."""
        placements = {}
        placements.update(self._placements_dict_from_object('agent'))
        for obstacle in self._obstacles:
            if obstacle.name != 'hazards':
                placements.update(self._placements_dict_from_object(obstacle.name))
        self.placements_conf.placements = placements

    def _build(self) -> None:
        """build_layout only samples agent/goal, and then writes the fixed hazard position into the layout."""
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        self.world_info.layout = self.random_generator.build_layout()
        for i, xy in enumerate(self._seg_locs):
            self.world_info.layout[f'hazard{i}'] = np.array(xy, dtype=np.float64)
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        if self.world is None:
            self.world = World(
                self.agent, self._obstacles, self.world_info.world_config_dict
            )
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)
        self.world_info.reset_layout = deepcopy(self.world_info.layout)

    def _build_world_config(self, layout):
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout['agent'],
            'geoms': {},
            'free_geoms': {},
            'mocaps': {},
        }
        world_config['agent_rot'] = float(self.agent.rot)
        world_config['task_name'] = 'Semicircle'

        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            if getattr(obstacle, 'name', None) == 'hazards' and hasattr(self, '_hazard_rots') and len(self._hazard_rots) == num:
                rots = [float(r) for r in self._hazard_rots]
            else:
                rots = self.random_generator.generate_rots(num)
            obstacle.process_config(world_config, layout, rots)

        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])
        return world_config

    def calculate_reward(self):
        reward = 0.0
        reward += self._reward_step
        dist_goal = self.dist_goal()
        # Rewards for getting closer to the target (distance reduction * scale)
        if self.last_dist_goal is not None:
            reward += (self.last_dist_goal - dist_goal) * self._reward_distance_scale
        self.last_dist_goal = dist_goal
        if self.goal_achieved:
            reward += self._reward_goal
        cost = self.calculate_cost()
        reward += cost.get('cost_hazards', 0.0) * self._reward_hit
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal_position(self) -> None:
        """Goal is fixed not to resample, use goal_xy() directly."""
        self.world_info.layout['goal'] = np.array(goal_xy(), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def calculate_cost(self):
        return super().calculate_cost()

    @property
    def goal_achieved(self):
        return self.dist_goal() <= self.goal.size


# ---------- v1: The track is widened, the rest is the same as Level0 ----------
from . import semicircle_map_v1


class SemicircleLevel1(BaseTask):
    """Upper semicircle track v1: Lane widening (LANE_HALF_WIDTH larger), the rest is the same as Level0."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        m = semicircle_map_v1
        self.agent.locations = [m.start_xy()]
        self.agent.rot = m.start_rot()
        self.agent.keepout = 0.0
        margin = m.RADIUS + m.LANE_HALF_WIDTH + 0.2
        self.placements_conf.extents = [-margin, -0.2, margin, margin]
        self.placements_conf.margin = 0.0
        gx, gy = m.goal_xy()
        self._add_geoms(Goal(size=m.GOAL_SIZE, keepout=m.GOAL_SIZE * 0.5, locations=[(gx, gy)]))
        seg_locs, self._hazard_rots = m.segment_center_locations()
        self._add_geoms(
            PathSegmentHazards(num=len(seg_locs), locations=seg_locs, keepout=0.0)
        )
        self._seg_locs = seg_locs
        self.last_dist_goal = None
        self.mechanism_conf.continue_goal = False
        self.reward_conf.reward_clip = None
        self._reward_goal = 10.0
        self._reward_hit = -1.0
        self._reward_step = -0.01
        self._reward_distance_scale = 0.5

    def _build_placements_dict(self) -> None:
        placements = {}
        placements.update(self._placements_dict_from_object('agent'))
        for obstacle in self._obstacles:
            if obstacle.name != 'hazards':
                placements.update(self._placements_dict_from_object(obstacle.name))
        self.placements_conf.placements = placements

    def _build(self) -> None:
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        self.world_info.layout = self.random_generator.build_layout()
        for i, xy in enumerate(self._seg_locs):
            self.world_info.layout[f'hazard{i}'] = np.array(xy, dtype=np.float64)
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        if self.world is None:
            self.world = World(
                self.agent, self._obstacles, self.world_info.world_config_dict
            )
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)
        self.world_info.reset_layout = deepcopy(self.world_info.layout)

    def _build_world_config(self, layout):
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout['agent'],
            'geoms': {},
            'free_geoms': {},
            'mocaps': {},
        }
        world_config['agent_rot'] = float(self.agent.rot)
        world_config['task_name'] = 'Semicircle'
        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            if getattr(obstacle, 'name', None) == 'hazards' and hasattr(self, '_hazard_rots') and len(self._hazard_rots) == num:
                rots = [float(r) for r in self._hazard_rots]
            else:
                rots = self.random_generator.generate_rots(num)
            obstacle.process_config(world_config, layout, rots)
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])
        return world_config

    def calculate_reward(self):
        reward = 0.0
        reward += self._reward_step
        dist_goal = self.dist_goal()
        if self.last_dist_goal is not None:
            reward += (self.last_dist_goal - dist_goal) * self._reward_distance_scale
        self.last_dist_goal = dist_goal
        if self.goal_achieved:
            reward += self._reward_goal
        cost = self.calculate_cost()
        reward += cost.get('cost_hazards', 0.0) * self._reward_hit
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal_position(self) -> None:
        m = semicircle_map_v1
        self.world_info.layout['goal'] = np.array(m.goal_xy(), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def calculate_cost(self):
        return super().calculate_cost()

    @property
    def goal_achieved(self):
        return self.dist_goal() <= self.goal.size


# ---------- v2: Same geometry as v0, but arc hazard sampling is denser ----------
from . import semicircle_map_v2


class SemicircleLevel2(BaseTask):
    """Upper semicircle track v2: The same width as v0, but the arc hazard is spread out by spacing."""

    _MAP = semicircle_map_v2

    def __init__(self, config) -> None:
        super().__init__(config=config)
        m = type(self)._MAP
        self.agent.locations = [m.start_xy()]
        self.agent.rot = m.start_rot()
        self.agent.keepout = 0.0
        margin = m.RADIUS + m.LANE_HALF_WIDTH + 0.2
        self.placements_conf.extents = [-margin, -0.2, margin, margin]
        self.placements_conf.margin = 0.0
        gx, gy = m.goal_xy()
        self._add_geoms(Goal(size=m.GOAL_SIZE, keepout=m.GOAL_SIZE * 0.5, locations=[(gx, gy)]))
        seg_locs, self._hazard_rots = m.segment_center_locations()
        self._add_geoms(
            PathSegmentHazards(num=len(seg_locs), locations=seg_locs, keepout=0.0)
        )
        self._seg_locs = seg_locs
        self.last_dist_goal = None
        self.mechanism_conf.continue_goal = False
        self.reward_conf.reward_clip = None
        self._reward_goal = 10.0
        self._reward_hit = -1.0
        self._reward_step = -0.01
        self._reward_distance_scale = 0.5

    def _build_placements_dict(self) -> None:
        placements = {}
        placements.update(self._placements_dict_from_object('agent'))
        for obstacle in self._obstacles:
            if obstacle.name != 'hazards':
                placements.update(self._placements_dict_from_object(obstacle.name))
        self.placements_conf.placements = placements

    def _build(self) -> None:
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        self.world_info.layout = self.random_generator.build_layout()
        for i, xy in enumerate(self._seg_locs):
            self.world_info.layout[f'hazard{i}'] = np.array(xy, dtype=np.float64)
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)
        if self.world is None:
            self.world = World(
                self.agent, self._obstacles, self.world_info.world_config_dict
            )
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)
        self.world_info.reset_layout = deepcopy(self.world_info.layout)

    def _build_world_config(self, layout):
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout['agent'],
            'geoms': {},
            'free_geoms': {},
            'mocaps': {},
        }
        world_config['agent_rot'] = float(self.agent.rot)
        world_config['task_name'] = 'Semicircle'
        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            if getattr(obstacle, 'name', None) == 'hazards' and hasattr(self, '_hazard_rots') and len(self._hazard_rots) == num:
                rots = [float(r) for r in self._hazard_rots]
            else:
                rots = self.random_generator.generate_rots(num)
            obstacle.process_config(world_config, layout, rots)
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])
        return world_config

    def calculate_reward(self):
        reward = 0.0
        reward += self._reward_step
        dist_goal = self.dist_goal()
        if self.last_dist_goal is not None:
            reward += (self.last_dist_goal - dist_goal) * self._reward_distance_scale
        self.last_dist_goal = dist_goal
        if self.goal_achieved:
            reward += self._reward_goal
        cost = self.calculate_cost()
        reward += cost.get('cost_hazards', 0.0) * self._reward_hit
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal_position(self) -> None:
        m = type(self)._MAP
        self.world_info.layout['goal'] = np.array(m.goal_xy(), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def calculate_cost(self):
        return super().calculate_cost()

    @property
    def goal_achieved(self):
        return self.dist_goal() <= self.goal.size


class SemicircleLevel3(SemicircleLevel2):
    """Upper Semicircle v3: Same as v2, but adjust lidar maximum detection distance to 5."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.lidar_conf.max_dist = 5.0


class SemicircleLevel4(SemicircleLevel3):
    """Upper semicircle v4: Same as v3, but the goal is reduced in half and attached to the inner ring at the midpoint of the arc."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        m = type(self)._MAP
        self.goal.size = float(self.goal.size) * 0.4
        self._radius = float(m.RADIUS)
        self._lane_half_width = float(m.LANE_HALF_WIDTH)

    def build_goal_position(self) -> None:
        # Stick to the inner ring in the direction of the midpoint of the upper semicircle (theta=pi/2):
        # center_r = (R - lane_half_width) + goal_radius
        goal_y = (self._radius - self._lane_half_width) + float(self.goal.size)
        self.world_info.layout['goal'] = np.array((0.0, goal_y), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][0] = float(self.goal.size)
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][1] = float(self.goal.size) / 2.0
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)


class SemicircleLevel5(SemicircleLevel4):
    """Upper semicircle track v5: goal is GOAL_SIZE full size, attached to the inner ring, located 5/8 along the arc from the starting point to the end point; the distance shaping coefficient is 0.5."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        # safety_gymnasium Builder truncates when self.steps >= self.task.num_steps (see builder.py),
        # BaseTask defaults to num_steps=1000; it must be consistent with max_episode_steps=1500 of v5 in __init__.py.
        self.num_steps = 1500
        m = type(self)._MAP
        # Directly based on original GOAL_SIZE, set to 1.0x.
        self.goal.size = float(GOAL_SIZE) * 1.0
        # Restore distance shaping to 0.5 in v3
        self._reward_distance_scale = 0.5
        self._radius = float(m.RADIUS)
        self._lane_half_width = float(m.LANE_HALF_WIDTH)

    def build_goal_position(self) -> None:
        # The centerline polar angle decreases along the upper arc from the starting point pi to the end point 0; at arc length ratio s theta = pi * (1 - s). s=5/8 => theta=3pi/8.
        s_arc = 5.0 / 8.0
        theta = np.pi * (1.0 - s_arc)
        center_r = (self._radius - self._lane_half_width) + float(self.goal.size)
        goal_x = center_r * float(np.cos(theta))
        goal_y = center_r * float(np.sin(theta))
        self.world_info.layout['goal'] = np.array((goal_x, goal_y), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][0] = float(self.goal.size)
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][1] = float(self.goal.size) / 2.0
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)


# ---------- v6: v5 alignment + doubled lane width, goal world coordinates are the same as v5 ----------
from . import semicircle_map_v6


class SemicircleLevel6(SemicircleLevel5):
    """Upper semicircle track v6: Aligned with v5 (rewards, steps, lidar, goal position), the lane half-width is 2 times that of v5."""

    _MAP = semicircle_map_v6

    def __init__(self, config) -> None:
        super().__init__(config=config)
        m = type(self)._MAP
        self._radius = float(m.RADIUS)
        self._lane_half_width = float(m.LANE_HALF_WIDTH)

    def build_goal_position(self) -> None:
        gx, gy = semicircle_map_v6.goal_xy_v5_aligned()
        self.world_info.layout['goal'] = np.array((gx, gy), dtype=np.float64)
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][0] = float(self.goal.size)
        self.world_info.world_config_dict['geoms']['goal']['geoms'][0]['size'][1] = float(self.goal.size) / 2.0
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)
