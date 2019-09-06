from gym.envs.registration import register
import os.path as osp


register(
    id='FetchPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_fetchpush:FetchPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/FetchPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='ResidualPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'randomizer/config/ResidualPushRandomized/random.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='ResidualPushDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'randomizer/config/ResidualPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='TwoFrameResidualHookNoisyRandomizedEnv-v0',
    entry_point='randomizer.randomized_residual_hook:TwoFrameResidualHookNoisyEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/ResidualFetchHookRandomized/random.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='TwoFrameResidualHookNoisyDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_hook:TwoFrameResidualHookNoisyEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/ResidualFetchHookRandomized/default.json',
        'xml_name': 'hook.xml'
    }
)
