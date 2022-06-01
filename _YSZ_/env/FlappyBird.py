import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import FlappyBird as Env_FB


class FlappyBird(Env):
    # 如果想把画面渲染出来，就传参display_screen=True
    def __init__(self, **kwargs):
        super(FlappyBird, self).__init__()
        self.display_screen = False  # 渲染画面
        self.game = Env_FB()
        self.p = PLE(self.game, **kwargs, display_screen=self.display_screen)
        self.action_set = self.p.getActionSet()

        self.reward_limit = 30
        self.cur_total_reward = 0

        # 4个输入状态：见函数self._get_obs
        self.observation_space = 4
        # 两个输出状态：跳或者不跳
        self.action_space = 2
        self.env_action_shape = 0

    def _get_obs(self):
        # 获取游戏的状态
        state = self.game.getGameState()
        pipe_gap = state["next_pipe_bottom_y"] - state["next_pipe_top_y"]
        # 小鸟与它前面一对水管中下面那根水管的水平距离
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        # 小鸟与它前面一对水管中下面那根水管的顶端的垂直距离
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        # 获取小鸟的水平速度
        velocity = state['player_vel']
        # 将这些信息封装成一个数据返回
        return np.array([dist_to_pipe_horz, dist_to_pipe_bottom, velocity, pipe_gap])

    def reset(self):
        self.cur_total_reward=0
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        self.cur_total_reward += reward
        if self.cur_total_reward >= self.reward_limit:
            done = True
        return obs, reward, done, dict()

    def seed(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        pass
