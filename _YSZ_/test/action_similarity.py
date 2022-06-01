import numpy as np


def get_actions_using_states(states, model):
    """
    根据state序列获得action序列
    Returns: action 序列
    -------
    """
    action_list = []
    for s in states:
        action_list.append(model.forward(s, train=False))
    return action_list


def MAE(Model_1_actions, Model_2_actions):
    """
    # 输入：DQN和Mimic模型的动作值，两者长度相等
    # 输出：MAE
    """
    if len(Model_1_actions) != len(Model_2_actions):  # 如果两者长度不相等，抛异常
        raise RuntimeError("list长度不相等")
    x = np.array(Model_1_actions)
    y = np.array(Model_2_actions)
    MAE = np.sum(np.abs(x - y)) / len(x)
    return MAE


def RMSD(Model_1_actions, Model_2_actions):
    """
    # 输入：DQN和Mimic模型的动作值，两者长度相等
    # 输出：RMSD
    """
    if len(Model_1_actions) != len(Model_2_actions):  # 如果两者长度不相等，抛异常
        raise RuntimeError("list长度不相等")
    x = np.array(Model_1_actions)
    y = np.array(Model_2_actions)
    RMSD = np.power(np.sum(np.power(x - y, 2)) / len(x), 0.5)
    return RMSD


def Accuracy(Model_1_actions, Model_2_actions):
    """
    # 输入：DQN和Mimic模型的动作值，两者长度相等
    # 输出：ACC
    """
    x = np.array(Model_1_actions)
    y = np.array(Model_2_actions)
    ACC = np.sum(x == y) / len(Model_1_actions)
    return ACC
