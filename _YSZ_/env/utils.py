import _YSZ_ as ysz

def choose_env(env_name):
    """环境选择器"""
    if env_name == "PredatorPrey":
        env = ysz.env.PredatorPrey.PredatorPrey()
        num_actions = env.action_space
        num_states = env.observation_space
        env_a_shape = env.env_action_shape
    elif env_name == "MountainCar-v0":
        env = ysz.env.MountainCar.MountainCar(version=env_name)
        num_actions = env.action_space
        num_states = env.observation_space
        env_a_shape = env.env_action_shape
    elif env_name == "FlappyBird":
        env = ysz.env.FlappyBird.FlappyBird()
        num_actions = env.action_space
        num_states = env.observation_space
        env_a_shape = env.env_action_shape
    elif env_name == "CartPole-v0":
        env = ysz.env.CartPole.CartPole(version=env_name)
        num_actions = env.action_space
        num_states = env.observation_space
        env_a_shape = env.env_action_shape
    else:
        raise Exception('环境名输入错误: {}'.format(env_name))
    return env, num_states, num_actions, env_a_shape