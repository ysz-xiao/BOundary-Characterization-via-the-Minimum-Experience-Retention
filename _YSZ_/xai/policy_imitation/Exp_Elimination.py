import numpy as np
from multiprocessing import Process, Lock


def exp_elimination(EXP):
    """
    经验削减算法
    Parameters
    ----------
    EXP：削减前的经验

    Returns：削减后的经验
    -------

    """
    """公共数据"""
    input_dim = len(EXP[0]) - 1
    old_count = len(EXP)
    EXP = np.unique(EXP, axis=0)  # 去除重复元素
    i = 0
    while i < len(EXP):
        if _is_surrunded(EXP, EXP[i], input_dim):
            EXP = np.delete(EXP, i, axis=0)
        else:
            i += 1
        print("\texp remove:" + str(old_count) + "->" + str(len(EXP)) + "               ", end="\r")
    print("\texp remove:" + str(old_count) + "->" + str(len(EXP)) + "               ")
    return EXP


def collect_exp_with_EE(MAX_EXP_COUNT, agent, env, max_step):
    """
    最优经验收集
    Parameters
    ----------
    MAX_EXP_COUNT：最大经验量
    agent：Teacher智能体
    env：环境
    max_step：单轮交互最大步数

    Returns：收集好的经验
    -------

    """
    EXP = []
    eps = 0
    while len(EXP) < MAX_EXP_COUNT and eps < MAX_EXP_COUNT:
        state = env.reset()
        for t in range(max_step):
            action = agent.choose_action(state, train=False)
            next_state, reward, done, _ = env.step(action)
            exp = []
            for s in state: exp.append(s)
            exp.append(action)
            EXP.append(exp)

            if done:
                eps += 1
                break
            state = next_state

        # 当经验收集到最大值，进行一次约简
        if len(EXP) > MAX_EXP_COUNT:
            EXP = list(exp_elimination(EXP))
        print("\t collect EXP:" + str(len(EXP)) + "/" + str(MAX_EXP_COUNT) + "             ", end="\r")
    print("\t collect EXP:" + str(len(EXP)) + "/" + str(MAX_EXP_COUNT) + "             ")
    return EXP


def _distance(x1, x2):
    """计算两点距离"""
    return np.linalg.norm(np.array(x1) - np.array(x2))


def _is_surrunded(EXP, exp, dim):
    """
    判断当前点是否被相关经验包裹
    Parameters
    ----------
    EXP：经验池
    exp：待判断经验
    dim：状态维度

    Returns
    -------

    """
    # step1:找到p最近的异分类点p'
    exp_, dis_ = EXP[0], float('inf')  # 最近异分类点，距离
    for i in range(len(EXP)):
        if (exp[dim] != EXP[i][dim]):
            dis_tmp = _distance(EXP[i][0:dim], exp[0:dim])
            if dis_tmp < dis_:
                exp_ = EXP[i]
                dis_ = dis_tmp
    # step2:找出P中与当前点p距离不超过dis的点的集合
    for i in range(len(EXP)):
        if exp[dim] == EXP[i][dim]:
            dis1 = _distance(exp[0:dim], EXP[i][0:dim])
            if dis1 >= dis_:
                continue
            dis2 = _distance(exp_[0:dim], EXP[i][0:dim])
            if dis2 >= dis_:
                continue
            return True  # 判断为中间点
    return False  # 判定为边界点
