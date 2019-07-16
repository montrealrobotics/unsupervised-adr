from gym.envs.registration import register


register(
    id='ResidualFetchPickAndPlace-v0',
    entry_point='rpl_environments.envs:ResidualFetchPickAndPlaceEnv',
    max_episode_steps=50,
)

register(
    id='ResidualSlipperyPush-v0',
    entry_point='rpl_environments.envs:ResidualSlipperyPushEnv',
    max_episode_steps=50,
)

register(
    id='SlipperyPush-v0',
    entry_point='rpl_environments.envs:SlipperyPushEnv',
    max_episode_steps=50,
)

register(
    id='FetchHook-v0',
    entry_point='rpl_environments.envs:FetchHookEnv',
    max_episode_steps=100,
)

register(
    id='ResidualFetchHook-v0',
    entry_point='rpl_environments.envs:ResidualFetchHookEnv',
    max_episode_steps=100,
)

register(
    id='TwoFrameResidualHookNoisy-v0',
    entry_point='rpl_environments.envs:TwoFrameResidualHookNoisyEnv',
    max_episode_steps=100,
)

register(
    id='TwoFrameHookNoisy-v0',
    entry_point='rpl_environments.envs:TwoFrameHookNoisyEnv',
    max_episode_steps=100,
)

register(
    id='ResidualMPCPush-v0',
    entry_point='rpl_environments.envs:ResidualMPCPushEnv',
    max_episode_steps=50,
)

register(
    id='MPCPush-v0',
    entry_point='rpl_environments.envs:MPCPushEnv',
    max_episode_steps=50,
)

register(
    id='OtherPusherEnv-v0',
    entry_point='rpl_environments.envs:PusherEnv',
    max_episode_steps=150,
)

register(
    id='ResidualOtherPusherEnv-v0',
    entry_point='rpl_environments.envs:ResidualPusherEnv',
    max_episode_steps=150,
)

register(
    id='ComplexHook-v0',
    entry_point='rpl_environments.envs:ComplexHookEnv',
    max_episode_steps=100,
)

register(
    id='ResidualComplexHook-v0',
    entry_point='rpl_environments.envs:ResidualComplexHookEnv',
    max_episode_steps=100,
)

