import numpy as np


class SVM:

    def __init__(self, kernel, C, sigma=None, p=None) -> None:
        self.alpha = None
        self.Y = None
        self.X = None
        self.kernel = kernel
        self.b = None
        self.C = C
        self.p = p
        self.sigma = sigma
        self.max_iter = 300
        self.idY = None
        self.Yid = None

    def load_data(self, data: np.ndarray):
        y = data[:, 0]
        x = data[:, 1:]
        labels = np.unique(y)
        idY = {0: labels[0], 1: labels[1]}
        Yid = {labels[0]: 0, labels[1]: 1}
        y = (y == labels[1]).astype(int)
        self.X = x
        self.Y = y
        self.idY = idY
        self.Yid = Yid

    def fit(self):
        # 初始化
        self._initialize()
        k = 0
        b = 0
        while k < self.max_iter:
            for i in range(len(self.X)):
                # 选择两点进行优化
                a_i, x_i, y_i = self.alpha[i], self.X[i], self.Y[i]
                j = self.select_j(i)
                a_j, x_j, y_j = self.alpha[j], self.X[j], self.Y[j]
                # 计算核函数和期望等
                Kii = self.kernel(x_i, x_i)
                Kjj = self.kernel(x_j, x_j)
                Kij = self.kernel(x_i, x_j)
                eta = Kii + Kjj - 2 * Kij
                Ei = self.forward(x_i) - y_i
                Ej = self.forward(x_j) - y_j
                # 计算上下界
                L = np.max([0, a_j - a_i])
                H = np.min([self.C, self.C + a_j - a_i])
                # 计算新的a_j, a_i
                a_j_new = a_j + y_j * (Ei - Ej) / eta
                if a_j_new > H:
                    a_j_new = H
                elif a_j_new < L:
                    a_j_new = L
                a_i_new = a_i + y_i * y_j * (a_j - a_j_new)
                if abs(a_j_new - a_j) < 0.00001:
                    continue
                index = (self.alpha > 0) * (self.alpha < self.C)
                index = np.arange(len(self.alpha))[index]
                if len(index) == 0:
                    continue
                else:
                    index = index[0]
                    b = self.Y[index] - self.forward(self.X[index])
                # 更新 alpha 和 b
                self.alpha[i] = a_i_new
                self.alpha[j] = a_j_new
                self.b = b
            k += 1


    def select_j(self, i):
        index = np.arange(len(self.X))
        index = index[index != i]
        return np.random.choice(index)
    
    def forward(self, x):
        value = 0
        for j, z in enumerate(self.X):
            value += self.alpha[j] * self.Y[j] * self.kernel(z, x)
        return value + self.b

    def _initialize(self):
        self.alpha = np.zeros(self.X.shape[0])

    def _poly_kernel(self, x, z):
        return (x @ z + 1) ** self.p
    
    def _gaussian_kernel(self, x, z):
        return np.exp(-np.linalg.norm(x - z) / (2 * self.sigma ** 2))
    
    @staticmethod
    def _linear(x, z):
        return x @ z