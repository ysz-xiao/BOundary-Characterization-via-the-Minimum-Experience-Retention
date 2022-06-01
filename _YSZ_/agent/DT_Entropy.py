from sklearn import tree


class DT_Entropy():
    """Decision Tree: CART"""

    def __init__(self, max_depth, criterion='entropy'):
        super(DT_Entropy, self).__init__()
        self.CRITERION = criterion
        self.MAX_DEPTH = max_depth
        self.model = tree.DecisionTreeClassifier(criterion=self.CRITERION, max_depth=self.MAX_DEPTH)

    def save_to_desk(self, path, file_name):
        """存储模型"""
        import pickle
        with open(path + file_name + ".pickle", 'wb') as f:
            pickle.dump(self.model, f)

    def load_from_desk(self, path, file_name):
        """加载模型"""
        import pickle
        with open(path + file_name + ".pickle", 'rb') as f:
            self.model = pickle.load(f)

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
