from gym.envs.registration import register
import os.path as osp


register(
    id='FetchPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_fetchpush:FetchPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'randomizer/config/FetchPushRandomized/random.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='FetchSlideRandomizedEnv-v0',
    entry_point='randomizer.randomized_fetchslide:FetchSlideRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'randomizer/config/FetchSlideRandomized/random.json',
        'xml_name': 'slide.xml'
    }
)

register(
    id='FetchHookRandomizedEnv-v0',
    entry_point='randomizer.randomized_residual_hook:FetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/FetchHookRandomized/random.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='FetchHookDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_hook:FetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/FetchHookRandomized/default.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='NoisyFetchHookRandomizedEnv-v0',
    entry_point='randomizer.randomized_residual_hook:NoisyResidualFetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/FetchHookRandomized/random.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='NoisyFetchHookDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_hook:NoisyResidualFetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'randomizer/config/FetchHookRandomized/default.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='ErgoPushRandomizedEnv-Headless-v0',
    entry_point='randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'randomizer/config/ErgoPushRandomized/random.json',
        'headless': True
    }
)
register(
    id='ErgoPushDefaultEnv-Headless-v0',
    entry_point='randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'randomizer/config/ErgoPushRandomized/default.json',
        'headless': True
    }
)
register(
    id='ErgoPushRandomizedEnv-Graphical-v0',
    entry_point='randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'randomizer/config/ErgoPushRandomized/random.json',
        'headless': False
    }
)

