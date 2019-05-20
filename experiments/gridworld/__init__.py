from gym.envs.registration import register


register(
    id='dense-v0',
    entry_point='gridworld.dense:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)

register(
    id='sparse-v0',
    entry_point='gridworld.sparse:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)

register(
    id='simple-v0',
    entry_point='gridworld.simple:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)
