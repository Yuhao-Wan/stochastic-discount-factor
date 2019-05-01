from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gridworld.maze:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)
