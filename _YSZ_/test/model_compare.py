import numpy
import _YSZ_ as ysz


def choose_best_model(model1, model2, env, test_time=100, max_step=1000):
    '''
    测试，返回结果回报更高的模型
    Parameters
    ----------
    model1  模型1
    model2  模型2
    env 环境
    test_time   测试次数
    max_step    单次最大步数

    Returns
    -------

    '''
    r_list_1, r_list_2 = [], []
    for t_time in range(test_time):  # 测试100次
        r_list_1.append(ysz.test.generator_episode.episode_reward(env, model1, max_step=max_step, train=False))
        r_list_2.append(ysz.test.generator_episode.episode_reward(env, model2, max_step=max_step, train=False))
    avgR_1, avgR_2 = numpy.mean(r_list_1), numpy.mean(r_list_2)
    if avgR_1 < avgR_2:  # 如果当前模型超过历史，就存储
        print("\tbest model: {}-->{}".format(avgR_1, avgR_2))
        best_model, best_reward = model2, avgR_2
    else:
        best_model, best_reward = model1, avgR_1
    return best_model, best_reward
