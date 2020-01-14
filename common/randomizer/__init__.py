from gym.envs.registration import register
import os.path as osp


register(
    id='FetchPushRandomizedEnv-v0',
    entry_point='common.randomizer.randomized_fetchpush:FetchPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'common/randomizer/config/FetchPushRandomized/random.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='FetchSlideRandomizedEnv-v0',
    entry_point='common.randomizer.randomized_fetchslide:FetchSlideRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'common/randomizer/config/FetchSlideRandomized/random.json',
        'xml_name': 'slide.xml'
    }
)

register(
    id='FetchHookRandomizedEnv-v0',
    entry_point='common.randomizer.randomized_residual_hook:FetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/FetchHookRandomized/random.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='FetchHookDefaultEnv-v0',
    entry_point='common.randomizer.randomized_residual_hook:FetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/FetchHookRandomized/default.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='NoisyFetchHookRandomizedEnv-v0',
    entry_point='common.randomizer.randomized_residual_hook:NoisyResidualFetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/FetchHookRandomized/random.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='NoisyFetchHookDefaultEnv-v0',
    entry_point='common.randomizer.randomized_residual_hook:NoisyResidualFetchHookEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/FetchHookRandomized/default.json',
        'xml_name': 'hook.xml'
    }
)
register(
    id='ErgoPushRandomizedEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/random.json',
        'headless': True
    }
)
register(
    id='ErgoPushDefaultEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/default.json',
        'headless': True
    }
)
register(
    id='ErgoPushHardEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/hard.json',
        'headless': True
    }
)
register(
    id='ErgoReacherRandomizedEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/random.json',
        'headless': True,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
register(
    id='ErgoReacherDefaultEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/default.json',
        'headless': True,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
register(
    id='ErgoReacherHardEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/hard.json',
        'headless': True,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
register(
    id='ErgoReacherDefaultEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/hard.json',
        'headless': False,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
register(
    id='ErgoPushRandomizedEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/random.json',
        'headless': False
    }
)
register(
    id='ErgoPushHardEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/hard.json',
        'headless': False
    }
)

register(
    id='ResFetchPushRandomizedEnv-v0',
    entry_point='common.randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'common/randomizer/config/FetchPushRandomized/random.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='ResFetchPushDefaultEnv-v0',
    entry_point='common.randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=50,
    kwargs={
        'config': 'common/randomizer/config/FetchPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)