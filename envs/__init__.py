from gym.envs.registration import register

from envs.config import CONFIG_PATH
from envs.lunar_lander import LunarLanderRandomized
import os.path as osp

# Needed because of gym.space error in normal LunarLander-v2
register(
    id='LunarLanderDefault-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/default.json'}
)

register(
    id='LunarLanderDebug-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/debug.json'}
)

register(
    id='LunarLanderHard-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/hard.json'}
)

register(
    id='LLUberLow-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_68.json'}
)

register(
    id='LLUberHigh-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_1820.json'}
)

register(
    id='LunarLanderRandomized-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_820.json'}
)

register(
    id='LL811-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_811.json'}
)

register(
    id='LL1316-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_1316.json'}
)

register(
    id='LL820-v0',
    entry_point='envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'envs/config/LunarLanderRandomized/random_820.json'}
)
