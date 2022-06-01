import numpy as np


class KNN_Force():
    """KNN Force searching"""

    def __init__(self):
        super(KNN_Force, self).__init__()

    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def save_to_desk(self, path, file_name):
        """存储模型"""
        import pickle
        with open(path + file_name + ".pickle", 'wb') as f:
            pickle.dump(self, f)

    def load_from_desk(self, path, file_name):
        """加载模型"""
        import pickle
        with open(path + file_name + ".pickle", 'rb') as f:
            self = pickle.load(f)

    def fit(self, X, Y):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        self.X = X
        self.Y = Y
        self.Exp_Count = len(X)
        return self

    def forward(self, x, train=False):
        """x is [[1]]"""
        nearest = [-1, float('inf')]
        for i in range(self.Exp_Count):
            cur_dis = self.distance(x, self.X[i])
            if cur_dis < nearest[1]:
                nearest[0] = i
                nearest[1] = cur_dis
        result = self.Y[nearest[0]]
        return result
