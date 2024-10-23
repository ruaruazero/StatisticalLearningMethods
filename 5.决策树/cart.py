import pandas as pd
import numpy as np


class Node:

    def __init__(self, 
                 pre=None,
                 left=None,
                 right=None,
                 attr=None,
                 value=None,
                 ) -> None:
        self.pre = pre
        self.left = left
        self.right = right
        self.attr = attr
        self.value = value
        self.is_leaf = False


class CART:

    def __init__(self, data:pd.DataFrame, target, threshold, classify=False) -> None:
        self.data = data
        self.target = target
        self.threshold = threshold
        self.classify = classify
        if classify:
            self.tree = self.regression(self.data)
        else:
            self.tree = self.classification(self.data)


    def regression(self, data:pd.DataFrame, node: Node=None):
        if node is None:
            node = Node()
        # 递归结束条件
        if len(data) < self.threshold:
            node.is_leaf = True
            node.value = data[self.target].mean()
            return node
        attr, value = self.search_attr_value(data)
        node.attr = attr
        node.value = attr
        ind = self.data[attr <= value]
        r1 = data[ind]
        r2 = data[~ind]
        left_node = node(pre=node)
        right_node = node(pre=node)
        left = self.regression(r1, left_node)
        right = self.regression(r2, right_node)
        node.left = left
        right = right
        return node

    def search_attr_value(self):
        m_ls = None
        b_attr = None
        b_value = None
        for c in self.data.columns:
            if c == self.target:
                continue
            tmp_data = self.data[c, self.target]
            c_data = self.data.groupby(c).count()
            values = c_data.index
            for v in values:
                if self.classify:
                    ind = tmp_data[c] == v
                    r1 = tmp_data[ind]
                    r2 = tmp_data[~ind]
                    ls = self.condition_gini(r1, r2)
                else:
                    ind = tmp_data[c] <= v
                    r1 = tmp_data[ind]
                    r2 = tmp_data[~ind]
                    ls = self.least_squares(r1, r2)
                if m_ls is None:
                    m_ls = ls
                    b_attr = c
                    b_value = v
                elif ls < m_ls:
                    m_ls = ls
                    b_attr = c
                    b_value = v
        return b_attr, b_value


    def least_squares(self, r1: pd.DataFrame, r2: pd.DataFrame):
        c1 = r1[self.target].mean()
        c2 = r2[self.target].mean()
        d1 = ((r1[self.target] - c1) ** 2).sum()
        d2 = ((r2[self.target] - c2) ** 2).sum()
        return d1 + d2
    
    def classification(self, data: pd.DataFrame, node:Node=None):
        if node is None:
            node = Node()
        if len(data) < self.threshold:
            gc = data.groupby(self.target).count()
            t = np.argmax(gc.iloc[:,0].to_numpy())
            node.value = list(gc.index)[t]
            node.is_leaf = True
            return node
        attr, value = self.search_attr_value(data)
        node.attr = attr
        node.value = attr
        ind = self.data[attr == value]
        r1 = data[ind]
        r2 = data[~ind]
        left_node = node(pre=node)
        right_node = node(pre=node)
        left = self.classification(r1, left_node)
        right = self.classification(r2, right_node)
        node.left = left
        right = right
        return node
        
    def condition_gini(self, r1, r2):
        len_d = len(r1) + len(r2)
        w1 = len(r1) / len_d
        w2 = len(r2) / len_d
        g1 = self.gini(r1)
        g2 = self.gini(r2)
        return w1 * g1 + w2 * g2

    def gini(self, d: pd.DataFrame):
        k = d.groupby(self.target).count()
        counts = k.iloc[:, 0].to_numpy()
        p = (counts / len(d)) ** 2
        return 1 - p.sum()

    def predict(self, test_x: pd.DataFrame):
        pre_y = []
        if self.classify:
            for index, line in test_x.iterrows():
                node = self.tree
                while True:
                    if node.is_leaf:
                        pre_y.append(node.value)
                        break
                    if line[node.attr] == node.value:
                        node = node.left
                    else:
                        node = node.right
            return np.array(pre_y)
        else:
            pass

    def pre_loss(self, tdata: pd.DataFrame):
        test_x = tdata.drop(self.target, inplace=False)
        test_y = tdata[self.target].to_numpy()
        pred_y = self.predict(test_x)
        # 此处为预测损失，

    def cutting_tree(self, tdata):
        pass


    