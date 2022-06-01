import random
from gym import spaces
import numpy as np


def visualization_maze_3d(EXP):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    colors = ['red', 'blue', 'green', 'orange']
    EXP_visual = np.unique(EXP, axis=0)  # 去除重复元素
    # 逐个统计数量
    Z = np.zeros(len(EXP_visual), dtype=np.int32)
    for i in range(len(EXP_visual)):
        for j in range(len(EXP)):
            if (EXP_visual[i] == EXP[j]).all():
                Z[i] += 1
    # 打开画图窗口1，在三维空间中绘图
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    for i in range(len(EXP_visual)):
        # 给出点（0，0，0）和（100，200，300）
        x = [EXP_visual[i][0], EXP_visual[i][0]]
        y = [EXP_visual[i][1], EXP_visual[i][1]]
        z = [0, Z[i]]
        print("{},{},{}".format(x, y, z))
        # 将数组中的前两个点进行连线
        ax.plot(x, y, z, linewidth=2, c=colors[np.int32(EXP_visual[i][2])])
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 250)
        ax.set_xlabel('s[0]')
        ax.set_ylabel('s[1]')
        ax.set_zlabel('count')
    plt.show()


class PredatorPrey:
    def __init__(self, Max_X=20, Max_Y=20):
        super(PredatorPrey, self).__init__()
        self.Max_X = Max_X
        self.Max_Y = Max_Y
        self.predator_location = [random.randint(0, self.Max_X), random.randint(0, self.Max_Y)]
        self.prey_location = [random.randint(0, self.Max_X), random.randint(0, self.Max_Y)]
        self.action_space = 4
        self.observation_space = 2
        self.env_action_shape = 0
        self.MAXIMUM_STEP = 500  # 最大步数，超过直接结束
        self.CURSTATE = 0

    def _getCurState(self):
        cur_state = [self.predator_location[0] - self.prey_location[0],
                     self.predator_location[1] - self.prey_location[1]]
        return cur_state

    def reset(self):
        self.__init__()
        return self._getCurState()

    def is_done(self):
        if self.CURSTATE >= self.MAXIMUM_STEP:
            return True
        else:
            self.CURSTATE += 1
            return False

    def step(self, action):
        # predator
        if action == 0:
            self.predator_location[0] -= 1
        elif action == 1:
            self.predator_location[0] += 1
        elif action == 2:
            self.predator_location[1] -= 1
        elif action == 3:
            self.predator_location[1] += 1

        # prey
        prey_action = random.randint(0, 4)
        if prey_action == 0:
            self.prey_location[0] -= 1
        elif prey_action == 1:
            self.prey_location[0] += 1
        elif prey_action == 2:
            self.prey_location[1] -= 1
        elif prey_action == 3:
            self.prey_location[1] += 1

        # location越界处理
        if self.predator_location[0] > self.Max_X:
            self.predator_location[0] = self.Max_X
        if self.predator_location[0] < 0:
            self.predator_location[0] = 0
        if self.predator_location[1] > self.Max_Y:
            self.predator_location[1] = self.Max_Y
        if self.predator_location[1] < 0:
            self.predator_location[1] = 0

        if self.prey_location[0] > self.Max_X:
            self.prey_location[0] = self.Max_X
        if self.prey_location[0] < 0:
            self.prey_location[0] = 0
        if self.prey_location[1] > self.Max_Y:
            self.prey_location[1] = self.Max_Y
        if self.prey_location[1] < 0:
            self.prey_location[1] = 0
        next_state = self._getCurState()

        if self.prey_location != self.predator_location and self.is_done() == False:
            reward = -1
            done = False
        else:
            reward = 0
            done = True
        return next_state, reward, done, "_"


class _KNN_Force():
    """KNN Force searching"""

    def __init__(self):
        super(_KNN_Force, self).__init__()

    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def fit(self, X, Y):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        self.X = X
        self.Y = Y
        self.Exp_Count = len(X)
        return self

    def forward(self, x):
        """x is [[1]]"""
        nearest = [-1, float('inf')]
        for i in range(self.Exp_Count):
            cur_dis = self.distance(x, self.X[i])
            if cur_dis < nearest[1]:
                nearest[0] = i
                nearest[1] = cur_dis
        result = self.Y[nearest[0]]
        return result
