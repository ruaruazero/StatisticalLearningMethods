import numpy as np
from sklearn.datasets import load_iris


class logisticRegression:

    def __init__(self) -> None:
        self.w = None
        self.X = None
        self.Y = None
        self.loop = 30
        self.lr = 0.1
        self.idY = None

    def _load_data(self, data: np.ndarray):
        x = data[:, :-1]
        y = data[:, -1].reshape(x.shape[0], 1)
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        self.X = x
        y_label = np.unique(y)
        if y_label.shape != (2,):
            raise ValueError("Y label must be 2")
        dic = {0: y_label[0], 1: y_label[1]}
        Yid = {y_label[0]: 0, y_label[1]: 1}
        y = np.array([Yid[i] for i in y.flatten()]).reshape(y.shape)
        self.idY = dic
        self.Y = y

    def _init_weights(self):
        self.w = np.random.uniform(size=(self.X.shape[1]))

    def _exp(self, x):
        # return np.exp(np.sum(self.w * x, axis=1)) 
        return np.exp(self.w * x) 
    
    def fit(self, data):
        self._load_data(data)
        self._init_weights()
        for _ in range(self.loop):
            for ind, x in enumerate(self.X):
                gred = self._gradient(x, self.Y[ind])
                new_w = self.w + self.lr * gred
                self.w = new_w

    def predict(self, data):
        exp = self._exp(data)
        p1 = exp / (1 + exp)
        p0 = 1 / exp
        label = (p1 > p0).astype(int)
        return np.array([self.idY[i] for i in label.flatten()])
        
    
    def _gradient(self, X, y):
        # gred = self.X * self.Y - (self._exp(self.X) / (1 + self._exp(self.X))).reshape(self.X.shape[0], 1)
        gred = X * y - (self._exp(X) / (1 + self._exp(X)))
        # gred = np.sum(self.w * gred, axis=0)
        return gred


class MaxEntropy:

    def __init__(self) -> None:
        pass

    def f(self):
        pass


if __name__ == "__main__":
    load = load_iris()
    x = load.data[:100,:2]
    y = load.target[:100].reshape(100, 1)
    data = np.concatenate([x, y], axis=1)
    lgr = logisticRegression()
    lgr.fit(data)
    print(1)