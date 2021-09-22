from gym.envs.registration import register

register(
    id='quadcopter-v0',
    entry_point='gym_quadcopter.envs:QuadcopterEnv',
)
