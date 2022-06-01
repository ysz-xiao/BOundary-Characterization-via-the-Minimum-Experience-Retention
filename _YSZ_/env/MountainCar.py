import gym



class MountainCar:
    def __init__(self, version="MountainCar-v0"):
        super(MountainCar, self).__init__()
        self.env = gym.make(version)
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.env_action_shape = 0 if isinstance(self.env.action_space.sample(), int) else self.env.action_space.sample.shape

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
