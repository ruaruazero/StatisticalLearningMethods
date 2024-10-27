import numpy as np


class Node:

    def __init__(self, attr=None, label=None) -> None:
        self.children = {}
        self.attr = attr
        self.label = label

    def add_child(self, n, v):
        self.children[v] = n


class BaseModel:

    def __init__(self) -> None:
        pass

    def fit(self, args):
        pass

    def forward(self, X) -> np.ndarray:
        pass


class DecisionStump(BaseModel):

    def __init__(self) -> None:
        self.root = None

    def fit(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray):
        a = self._info_gain(X, Y, W)
        attr_values = np.unique(X[:, a])
        index = np.array([X[:a] == v for v in attr_values])
        root = Node(attr=a)
        leaf_Y = np.array([Y[i] for i in index])
        for loc, leaf in enumerate(leaf_Y):
            y_values = np.unique(leaf)
            if y_values.shape[0] == 1:
                label = y_values[0]
            else:
                counts = np.array([(leaf == v).astype(int).sum() for v in y_values])
                label = y_values[np.argmax(counts)]
            n = Node(label=label)
            root.add_child(n, attr_values[loc])
        self.root = root

    
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = X.flatten()
        root = self.root
        attr = root.attr
        label = root.children[X[attr]].label
        return label

    
    def _info_gain(self, X:np.ndarray, Y:np.ndarray, W:np.ndarray)->int:
        e = self.entropy(Y, W)
        gains = np.array([])
        for c in range(X.shape[1]):
            D_a = X[:, c].flatten()
            values = np.unique(D_a)
            g = np.array([])
            for v in values:
                index = D_a == v
                r_v = W[index].sum() / W.sum()
                Y_v = Y[index]
                W_v = W[index]
                g = np.append(g, r_v * self.entropy(Y_v, W_v))
            gs = e - g.sum()
            gains = np.append(gains, gs)
        return np.argmax(gains)


    @staticmethod
    def entropy(Y: np.ndarray, W: np.ndarray):
        values = np.unique(Y)
        index = np.array([Y == v for v in values])
        s = W.sum()
        p = np.array([W[i].sum() for i in index])
        p = p / s
        return (-p * np.log2(p)).sum()


class AdaBoost:

    def __init__(self, base_model: BaseModel, max_iter) -> None:
        self.w = None
        self.X = None
        self.Y = None
        self.base_model = base_model
        self.max_iter = max_iter
        self.g_s = None
        self.a_s = None
        self.idY = None
        self.Yid = None

    def _init_weights(self):
        self.w = np.ones(self.X.shape[0]) / self.X.shape[0]

    def _load_data(self, X, Y):
        labels = np.unique(Y)
        idY = {-1: labels[0], 1: labels[1]}
        Yid = {labels[0]: -1, labels[1]: 1}
        y = (y == labels[1]).astype(int)
        y = y * 2 - 1
        self.X = X
        self.Y = y
        self.idY = idY
        self.Yid = Yid

    def fit(self, X, Y):
        g_s = []
        a_s = []
        self._load_data(X, Y)
        self._init_weights()
        for _ in range(self.max_iter):
            # 训练基分类器
            g_m = self.base_model()
            g_m = g_m.fit(self.X, self.Y, self.w)
            g_s.append(g_m)
            # 计算e_m
            p_m = g_m.forward(self.X)
            I = (p_m != self.Y).astype(int)
            e_m = (self.w * I).sum()
            # 计算g_m的系数
            a_m = (1 / 2) * np.log((1 - e_m) / e_m)
            a_s.append(a_m)
            # 更新权值
            w_m = self.w * np.exp(-a_m * self.Y * p_m)
            z_m = w_m.sum()
            new_w = w_m / z_m
            self.w = new_w
        self.g_s = g_s
        self.a_s = a_s
        
        
    def f(self, X):
        a = np.array(self.a_s)
        rs = [g.fit(X) for g in self.g_s]
        rs = np.array(rs)
        return (a * rs).sum()
    
    def forward(self, X):
        r = self.f(X)
        y = 0
        if r < 0:
            y = -1
        elif r > 0:
            y = 1
        return self.idY[y]
