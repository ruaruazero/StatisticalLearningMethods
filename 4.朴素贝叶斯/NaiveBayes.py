import pandas as pd


def naive_bayes(df: pd.DataFrame, cata: str, lam=0):
    N = df.shape[1]
    all_attr = df.index.to_list()
    all_values = {}
    shape = {}
    for attr in all_attr:
        tmp = df.loc[attr].value_counts().index.to_list()
        all_values[attr] = tmp
        shape[attr] = len(tmp)
    # 计算先验概率
    pre_prob = {}
    for value in all_values[cata]:
        n = df.loc[:, df.loc[cata] == value]
        pre_prob[value] = n.shape[1]
    # 准备概率矩阵
    prob_matrix = {}
    # 计算概率矩阵
    for attr in all_attr:
        if attr == cata:
            continue
        values = all_values[attr]
        tmp = {}
        for value in values:
            temp = {}
            for v in all_values[cata]:
                l1 = list(df.loc[cata] == v)
                l2 = list(df.loc[attr] == value)
                ind = [x and y for x, y in zip(l1, l2)]
                n = df.loc[:, ind].shape[1]
                temp[v] = (n + lam) / (pre_prob[v] + (shape[attr] * lam))
            tmp[value] = temp
        prob_matrix[attr] = tmp
    for key in pre_prob.keys():
        pre_prob[key] = (pre_prob[key] + lam) / (N + shape[cata] * lam)
    return pre_prob, prob_matrix


if __name__ == '__main__':
    data = pd.read_csv("/Users/0g/Documents/2425s/learning/统计学习算法/4.朴素贝叶斯/data.csv", header=0, index_col=0)
    naive_bayes(data, 'Y')