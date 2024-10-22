import numpy as np
import pandas as pd


class Node:
    
    def __init__(self, pre=None, attr=None, id=None) -> None:
        self.children = {}
        self.pre = pre
        self.cls = None
        self.attr = attr
        self.index = None
        self.id = id
        self.eh = None
        self.leaves_id = []
        self.is_leaf = False 
        self.leaf = []

    def add_child(self, child, value):
        self.children[value] = child
        if child.is_leaf:
            self.leaves_id.append(child.id)
        else:
            self.leaves_id += child.leaves_id

    def __str__(self) -> str:
        return f'attr: {self.attr}, cls: {self.cls}'
    
    def __repr__(self) -> str:
        return f'attr: {self.attr}, cls: {self.cls}'

def dict2tuple(dic):
    tmp = list(zip(dic.keys(), dic.values()))
    tmp.sort(key=lambda x: x[1], reverse=True)
    return tmp



def entropy(x):
    x = np.where(x == 0, 1, x)
    return (-x * np.log2(x)).sum()


def get_all_count(data: pd.DataFrame, target):
    res = {}
    features = list(data.columns)
    index = features.index(target)
    features.pop(index)
    target_count = data.groupby(target).count()
    target_values = list(target_count.index)
    res[target] = target_count.iloc[:, 0].to_numpy()
    for feature in features:
        d = data.groupby(feature).count()
        feature_values = list(d.index)
        arr = np.zeros((len(target_values), len(feature_values)))
        for i, tv in enumerate(target_values):
            tmp = data[data[target]==tv]
            for j, fv in enumerate(feature_values):
                foc = len(tmp[tmp[feature]==fv])
                arr[i, j] = foc
        res[feature] = arr
    return res


def information_gain(data: pd.DataFrame, target):
    all_count = get_all_count(data, target)
    hd = all_count[target]
    total = hd.sum()
    hd = hd / total
    hd = entropy(hd)
    res = {}
    for key in all_count.keys():
        if key == target:
            continue
        c = all_count[key]
        fc = c.sum(axis=0)
        fvp = c / fc
        fen = [entropy(fvp[:,i]) for i in range(c.shape[1])]
        fen = np.array(fen)
        ce = (fen * fc / total).sum()
        res[key] = hd - ce
    return dict2tuple(res)



def information_gain_rate(data: pd.DataFrame, target):
    all_count = get_all_count(data, target)
    hd = all_count[target]
    total = hd.sum()
    ig = information_gain(data, target)
    for key in ig.keys():
        i = ig[key]
        fc = all_count[key].sum(axis=0)
        fcr = fc / total
        hf = entropy(fcr)
        ig[key] = i / hf
    return dict2tuple(ig)



if __name__ == '__main__':
    d = pd.read_csv("/Users/0g/Documents/2425s/learning/StatisticalLearningMethods/5.决策树/data1.csv", header=0, index_col=0)
    t = information_gain(d, '类别')
    b = tuple(zip(t))
    print(b)
    print(t)
