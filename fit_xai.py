import argparse
import torch
import numpy as np
import _YSZ_ as ysz
from multiprocessing.pool import Pool
from multiprocessing import Process, Lock


def fit_xai_models(args):
    algorithms = []  # 算法标题，算法模型
    # 初始化
    env, num_states, num_actions, env_a_shape = ysz.env.utils.choose_env(args.env)  # 选择环境

    # DQN
    DQN = torch.load(args.teacher_path + args.env + ".model")  # 加载agent
    algorithms.append(["DQN", DQN])

    # collect exp
    EXP = ysz.interact.experience.collect_exp(DQN, env, args.max_exp_count,
                                              args.max_step)  # 收集经验

    # ---fit vanilla---
    X, Y = np.array(EXP)[:, 0:len(EXP[0]) - 1], np.array(EXP)[:, len(EXP[0]) - 1]  # 数据分割

    # DT entropy
    Vanilla_DT_Entropy_5 = ysz.agent.DT_Entropy.DT_Entropy(max_depth=5).fit(X, Y)
    algorithms.append(["Vanilla_DT_Entropy_5", Vanilla_DT_Entropy_5])
    Vanilla_DT_Entropy_5.save_to_desk(args.student_path, "Vanilla_DT_Entropy_5" + "_" + args.env + "_exp" + str(
        args.max_exp_count))
    Vanilla_DT_Entropy_10 = ysz.agent.DT_Entropy.DT_Entropy(max_depth=10).fit(X, Y)
    algorithms.append(["Vanilla_DT_Entropy_10", Vanilla_DT_Entropy_10])
    Vanilla_DT_Entropy_10.save_to_desk(args.student_path, "Vanilla_DT_Entropy_10" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    # DT gini
    Vanilla_Gini_5 = ysz.agent.DT_Gini.DT_Gini(max_depth=5).fit(X, Y)
    algorithms.append(["Vanilla_Gini_5", Vanilla_Gini_5])
    Vanilla_Gini_5.save_to_desk(args.student_path, "Vanilla_Gini_5" + "_" + args.env + "_exp" + str(
        args.max_exp_count))
    Vanilla_Gini_10 = ysz.agent.DT_Gini.DT_Gini(max_depth=10).fit(X, Y)
    algorithms.append(["Vanilla_Gini_10", Vanilla_Gini_10])
    Vanilla_Gini_10.save_to_desk(args.student_path, "Vanilla_Gini_10" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    # ---fit proposed---
    EXP_DEL = ysz.xai.policy_imitation.Exp_Elimination.exp_elimination(EXP)
    X_DEL, Y_DEL = np.array(EXP_DEL)[:, 0:len(EXP_DEL[0]) - 1], np.array(EXP_DEL)[:, len(EXP_DEL[0]) - 1]  # 数据分割

    # EE_Force
    EE_Force = ysz.agent.KNN_Force.KNN_Force().fit(X_DEL, Y_DEL)
    algorithms.append(["EE_Force", EE_Force])
    EE_Force.save_to_desk(args.student_path, "EE_Force" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    # EE_KD
    EE_KD = ysz.agent.KNN_KD.KNN_KD(n_neighbors=1).fit(X_DEL, Y_DEL)
    algorithms.append(["EE_KD", EE_KD])
    EE_KD.save_to_desk(args.student_path, "EE_KD" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    # EE_Ball
    EE_Ball = ysz.agent.KNN_Ball.KNN_Ball(n_neighbors=1).fit(X_DEL, Y_DEL)
    algorithms.append(["EE_Ball", EE_Ball])
    EE_Ball.save_to_desk(args.student_path, "EE_Ball" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    # EE_Brute
    EE_Brute = ysz.agent.KNN_Brute.KNN_Brute(n_neighbors=1).fit(X_DEL, Y_DEL)
    algorithms.append(["EE_Brute", EE_Brute])
    EE_Brute.save_to_desk(args.student_path, "EE_Brute" + "_" + args.env + "_exp" + str(
        args.max_exp_count))

    return [i[0] for i in algorithms], [i[1] for i in algorithms]


def test_reward(models, model_args, env):
    # test
    reward_list = []  # 本轮repeat的reward
    for repeat_i in range(model_args.repeat):  # repeat
        reward_tmp = []
        for alg_i in range(len(models)):  # 逐算法测试
            reward_tmp.append(ysz.test.generator_episode.episode_reward(env, models[alg_i], model_args.max_step))
        reward_list.append(reward_tmp)
        if repeat_i % 100 == 0:
            print("\trepeat {}/{}, rewards: {}".format(repeat_i, model_args.repeat, reward_tmp))
    return reward_list





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # "MountainCar-v0","PredatorPrey","FlappyBird","CartPole-v0"
    parser.add_argument('-env', '--env', default='PredatorPrey', type=str, help='gym environment')
    parser.add_argument('-t_path', '--teacher_path', default="./model/", type=str, help='')
    parser.add_argument('-s_path', '--student_path', default="./model/xai/", type=str, help='')
    parser.add_argument('-exp_c', '--max_exp_count', default=5000, type=int, help='exp_count')
    parser.add_argument('-exp_ce', '--max_collect_eps', default=50000, type=int, help='episode steps')
    parser.add_argument('-t_s', '--max_step', default=10000, type=int, help='单轮最大步数')
    parser.add_argument('-t_ep', '--repeat', default=1000, type=int, help='测试重复次数')
    args = parser.parse_args()

    # 初始化
    env, num_states, num_actions, env_a_shape = ysz.env.utils.choose_env(args.env)  # 选择环境
    do_similarity = True
    do_reward = True
    do_exp_number = True
    # 测试
    for exp_num in [500, 1000, 10000]:
        args.max_exp_count = exp_num
        print("------   fit xai model   ------")
        print("\texp={}, env={}".format(exp_num, args.env))
        model_names, models = fit_xai_models(args)  # models[0] is the teacher-DQN
        if do_reward:
            print("------   test rewards   ------")
            accumulate_rewards = test_reward(models, args, env)
            ysz.trainsform.data_storage.write_csv(model_names, accumulate_rewards,
                                                  "./result/accumulate_rewards/" + args.env + "_exp" + str(
                                                      args.max_exp_count) + ".csv", auto_cal=True)
