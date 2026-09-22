# Semicircle track environment: standard upper semicircle, from (-R,0) to (R,0)

import copy

from safety_gymnasium.utils import registration
import safety_gymnasium.tasks as _tasks
import safety_gymnasium.assets.geoms as _sg_geoms

from .semicircle_task import SemicircleLevel0, SemicircleLevel1, SemicircleLevel2, SemicircleLevel3, SemicircleLevel4, SemicircleLevel5, SemicircleLevel6
from .semicircle_assets import PathSegmentHazards

_sg_geoms.GEOMS_REGISTER.append(PathSegmentHazards)
_tasks.SemicircleLevel0 = SemicircleLevel0
_tasks.SemicircleLevel1 = SemicircleLevel1
_tasks.SemicircleLevel2 = SemicircleLevel2
_tasks.SemicircleLevel3 = SemicircleLevel3
_tasks.SemicircleLevel4 = SemicircleLevel4
_tasks.SemicircleLevel5 = SemicircleLevel5
_tasks.SemicircleLevel6 = SemicircleLevel6

PREFIX = 'Safety'


def _register():
    for version, task_name in [('v0', 'SemicircleLevel0'), ('v1', 'SemicircleLevel1'), ('v2', 'SemicircleLevel2'), ('v3', 'SemicircleLevel3'), ('v4', 'SemicircleLevel4'), ('v5', 'SemicircleLevel5'), ('v6', 'SemicircleLevel6')]:
        task_config = {'task_name': task_name}
        max_steps = 1500 if version == 'v5' else 1000
        for robot_name in ('Point', 'Car', 'Racecar'):
            env_id = f'{PREFIX}{robot_name}Semicircle0-{version}'
            config = copy.deepcopy(task_config)
            config['agent_name'] = robot_name
            registration.register(
                id=env_id,
                entry_point='safety_gymnasium.builder:Builder',
                kwargs={'config': config, 'task_id': env_id},
                max_episode_steps=max_steps,
            )


_register()

__all__ = ['SemicircleLevel0', 'SemicircleLevel1', 'SemicircleLevel2', 'SemicircleLevel3', 'SemicircleLevel4', 'SemicircleLevel5', 'SemicircleLevel6']
