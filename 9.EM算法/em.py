import numpy as np


class ProbModel:

    def __init__(self) -> None:
        pass

    def p():
        pass


class EM:

    def __init__(self) -> None:
        pass

    def fit(self, theta, num, Y, Z):
        theta = np.random.uniform(low=0, high=1, size=Y.shape[1] + 1)
        

class GaussianMixture(ProbModel):

    def __init__(self) -> None:
        pass

    def fit(self, Y, num):
        y = Y.reshape([-1, 1])
        y_ = np.concatenate([y]*num, axis=1)
        # 初始化参数
        alpha = np.random.uniform(low=0, high=1, size=num)
        alpha = alpha / alpha.sum()
        mu = np.random.uniform(size=num)
        sigma = np.random.uniform(size=num)
        while True:
            # 计算gamma
            phi = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp((- (y_ - mu) ** 2) / (2 * sigma ** 2)) # N * K
            all_gamma = alpha * phi
            gamma = all_gamma / all_gamma.sum(axis=1).reshape([-1, 1]) # N * K
            # 更新参数
            sum_gamma = gamma.sum(axis=0)
            new_mu = (gamma * y).sum(axis=0) / sum_gamma
            new_sigma = (gamma * (y - mu) ** 2).sum(axis=0) / sum_gamma
            new_alpha = sum_gamma / gamma.shape[0]
            # 确定是否收敛
            old_theta = np.concatenate([alpha, mu, sigma])
            new_theta = np.concatenate([new_alpha, new_mu, new_sigma])
            if np.linalg.norm(new_theta - old_theta) < 0.001:
                return new_alpha, new_mu, new_sigma
            else:
                alpha = new_alpha
                mu = new_mu
                sigma = new_sigma


if __name__ == '__main__':
    data = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
    gm = GaussianMixture()
    theta = gm.fit(data, 2)
    print(1)