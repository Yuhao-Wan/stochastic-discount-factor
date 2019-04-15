from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='gridworld.maze:MazeEnv',
    timestep_limit=10000,
    kwargs=dict(
    ),
    nondeterministic = False,
)
