import _YSZ_ as ysz
import argparse
import torch
import _YSZ_.agent.DQN as DQN
import numpy as np


def train(args):
    env, num_states, num_actions, env_a_shape = ysz.env.utils.choose_env(args.env)  # 环境选择
    model = DQN.DQN(num_states, num_actions, env_a_shape)
    torch.save(model, args.model_path + args.env + ".model")  # 存储初始模型
    best_model, best_R = None, -float('inf')
    # 训练
    for i in range(args.episode):
        state = env.reset()
        model.episilo *= model.episilo_decay
        for t in range(args.step):
            action = model.forward(state, train=True)
            next_state, reward, done, _ = env.step(action)
            model.store_transition(DQN.Transition(state, action, reward, next_state, done))
            model.learn()
            if done: break
            state = next_state

        # 与原始模型比较，挑选最好的模型存储
        if i % int(args.episode / 100) == 0:
            old_model = torch.load(args.model_path + args.env + ".model")  # 加载存储的模型
            best_model, best_R = ysz.test.model_compare.choose_best_model(old_model, model, env, max_step=args.step)
            torch.save(best_model, args.model_path + args.env + ".model")
            print("{} episode: {}/{}({:.1f}%), ""avg reward is {:.2f}".
                  format(args.env, i, args.episode, i / args.episode * 100, best_R))
    return best_model, best_R


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # PredatorPrey, MountainCar-v0, FlappyBird, CartPole-v0
    parser.add_argument('-env', '--env', default='CartPole-v0', type=str)
    parser.add_argument('-step', '--step', default=10000, type=int, help='episode steps')
    parser.add_argument('-ep', '--episode', default=10000, type=int, help='episode nums')
    parser.add_argument('-path', '--model_path', default='./model/', type=str)
    args = parser.parse_args()
    print("------ train {}  ------".format(args.env))

    # 开始训练
    train(args)
