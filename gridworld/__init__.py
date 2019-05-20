from gym.envs.registration import register


register(
    id='dense',
    entry_point='gridworld.dense:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)

register(
    id='sparse',
    entry_point='gridworld.sparse:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)


register(
    id='simple',
    entry_point='gridworld.simple:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)
