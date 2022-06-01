from sklearn.neighbors import KNeighborsClassifier


class KNN_Ball():
    """KNN Ball-Tree"""

    # algorithm={‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    def __init__(self, n_neighbors, algorithm='ball_tree'):
        super(KNN_Ball, self).__init__()
        self.N_NEIGHBORS = n_neighbors
        self.ALGORITHM = algorithm
        self.model = KNeighborsClassifier(n_neighbors=self.N_NEIGHBORS, algorithm=self.ALGORITHM)

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
    def fit(self, X, y):
        """
        X is [[1]...[N]]
        Y is [1...N]
        """
        self.model.fit(X, y)
        return self

    def forward(self, x,train=False):
        """x is [[1]]"""
        result = self.model.predict([x])[0]
        return result
