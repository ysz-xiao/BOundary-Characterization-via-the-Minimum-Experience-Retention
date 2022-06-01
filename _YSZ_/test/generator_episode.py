import numpy as np


def episode_state_actions(env, model, max_step, train=False):
    """一次episode的状态动作序列"""
    state_list, action_list = [], []
    state = env.reset()
    for step in range(max_step):
        action = np.int32(model.forward(state, train))
        state_list.append(state)
        action_list.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
    return state_list, action_list


def episode_reward(env, model, max_step, train=False):
    """一次episode的回报"""
    result_reward = 0
    state = env.reset()
    for step in range(max_step):
        action = np.int32(model.forward(state, train))

        next_state, reward, done, _ = env.step(action)
        result_reward += reward
        state = next_state
        if done:
            break
    return result_reward


def episode_predict_time(env, model, max_episode=100):
    """一次episode的平均决策时间"""
    import time
    import numpy
    time_list = []
    state = env.reset()
    for step in range(max_episode):
        start_time = time.time()
        action = model.forward(state)
        end_time = time.time()
        time_list.append(end_time - start_time)
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break
    avg_time = numpy.mean(time_list)
    return avg_time
