from gym.envs.registration import register
import os.path as osp

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
    id='ErgoPushImpossibleEnv-Headless-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': 'common/randomizer/config/ErgoPushRandomized/impossible.json',
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
    id='ErgoPushHardEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': '/home/sharath/unsupervised-adr/common/randomizer/config/ErgoPushRandomized/hard.json',
        'headless': False
    }
)
register(
    id='ErgoPushDefaultEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergo_pusher:ErgoPusherRandomizedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        'config': '/home/sharath/unsupervised-adr/common/randomizer/config/ErgoPushRandomized/default.json',
        'headless': False
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
    id='ErgoReacherDefaultEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/default.json',
        'headless': False,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
register(
    id='ErgoReacherHardEnv-Graphical-v0',
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
    id='ErgoReacherRanomizedEnv-Graphical-v0',
    entry_point='common.randomizer.randomized_ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': 'common/randomizer/config/ErgoReacherRandomized/random.json',
        'headless': False,
        'simple': True,
        'goal_halfsphere': True,
        'multi_goal': True
    }
)
