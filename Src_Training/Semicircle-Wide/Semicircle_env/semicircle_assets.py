# Semicircular track: multiple small rectangular hazard segments are spliced ​​into boundary walls along both sides of the arc

from dataclasses import dataclass

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom

from .semicircle_map import (
    SEGMENT_HALF_X,
    SEGMENT_HALF_Y,
    SEGMENT_HALF_Z,
)


@dataclass
class PathSegmentHazards(Geom):
    """Multiple segments of small rectangular hazards are spliced ​​together to form a boundary along the left and right sides of the semicircular arc."""

    name: str = 'hazards'
    num: int = 1
    placements: list = None
    locations: list = None
    keepout: float = 0.05
    alpha: float = 0.25
    cost: float = 1.0

    color: np.ndarray = COLOR['hazard']
    group: np.array = GROUP['hazard']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, SEGMENT_HALF_Z + 1e-3],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([SEGMENT_HALF_X, SEGMENT_HALF_Y, SEGMENT_HALF_Z]),
                    'type': 'box',
                    'contype': 1,
                    'conaffinity': 1,
                    'group': self.group,
                    'rgba': np.r_[np.asarray(self.color).flat[:3], self.alpha],
                },
            ],
        }
        return body

    def cal_cost(self):
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_hazards'] = 0.0
        hazard_geom_names = {f'{self.name[:-1]}{i}' for i in range(self.num)}
        agent_geom_names = set(self.agent.body_info.geom_names)
        for i in range(self.engine.data.ncon):
            c = self.engine.data.contact[i]
            g1 = self.engine.model.geom(c.geom1).name
            g2 = self.engine.model.geom(c.geom2).name
            if (g1 in agent_geom_names and g2 in hazard_geom_names) or (
                g2 in agent_geom_names and g1 in hazard_geom_names
            ):
                cost['cost_hazards'] += self.cost
                break
        return cost

    @property
    def pos(self):
        return [
            self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy()
            for i in range(self.num)
        ]
