import pandas as pd
import numpy as pd
import copy
from utils import *


def group_count(data, column):
    return data.groupby(column).count()


class DecisionTree:

    def __init__(self, data: pd.DataFrame, target, threshold, ig) -> None:
        self.data = data
        self.target = target
        self.threshold = threshold
        self.ig = ig
        self.leaf = []
        self.node_num = 0
        self.tree = self.build()
        self.tree.leaf = self.leaf


    def build(self):
        this = Node(id=self.node_num)
        return self.iteration(this, self.data)

    def iteration(self, node: Node, data: pd.DataFrame):
        gc = group_count(data, self.target)
        # 计算当前节点的经验熵
        nk = gc.iloc[:, 0].to_numpy()
        eh = entropy(nk / nk.sum())
        node.eh = eh
        # 判断是否所有实例属于一类
        if len(group_count(data, self.target)) == 1:
            node.cls = list(gc.index)[0]
            node.index = data.index
            node.is_leaf = True
            self.leaf.append(node)
            return node
        attrs = list(data.columns)
        # 判断是否无特征值 
        if len(attrs) == 1:
            if attrs[0] == self.target:   
                t = np.argmax(gc.iloc[:,0].to_numpy())
                node.cls = list(gc.index)[t]
                node.index = data.index
                node.is_leaf = True
                self.leaf.append(node)
                return node
            else:
                raise AttributeError("There is not such a target in data!")
        # 计算信息增益
        ig = self.ig(data, self.target)
        t = np.argmax(gc.iloc[:,0].to_numpy())
        node.cls = list(gc.index)[t]
        node.index = data.index
        if ig[0][1] < self.threshold:
            # 如果信息增益小于阈值
            node.is_leaf = True
            self.leaf.append(node)
            return node
        else:
            # 若信息增益大于阈值
            attr = ig[0][0]
            attr_values = list(group_count(data, attr).index)
            for value in attr_values:
                value_data = data[data[attr]==value]
                value_data = value_data.drop(columns=[attr])
                self.node_num += 1
                child_node = Node(pre=node, id=self.node_num)
                child_node = self.iteration(child_node, value_data)
                node.attr = attr
                node.add_child(child_node, value)
            return node
        
    def forward(self):
        pass

    def cutting_tree(self, alpha):
        self.tree = self.cutting_tree_iteration(self.tree, alpha)

    def cutting_tree_iteration(self, tree: Node, alpha, best_tree=None, ctm=None):
        """
        已知每个节点的经验熵，需从叶节点向上回缩
        leaf表示其中一个叶节点
        """
        if best_tree is None:
            best_tree = tree
            tmp_leaves = best_tree.leaf
            ctm = sum([n.eh for n in tmp_leaves]) + alpha * len(tmp_leaves)
        for leaf in tree.leaf:
            new_tree = copy.deepcopy(tree)
            new_leaves = new_tree.leaf
            leaf = [n for n in new_leaves if n.id == leaf.id][0]
            # 剪枝算法, 使用递归
            # 计算剪枝前的损失
            ctb = sum([n.eh for n in new_leaves]) + alpha * len(new_leaves)
            # 计算剪枝后的叶节点数量
            # 分为两种情况：
            # 1.父节点的另一子节点为叶节点
            # 2.父节点的另一子节点不是叶节点
            parent = leaf.pre
            # 此时已剪枝到根节点
            if parent is None:
                return best_tree
            # 剪枝
            delet_leaf = parent.leaves_id
            new_leaves = [n for n in new_leaves if n.id not in delet_leaf]
            parent.is_leaf = True
            parent.children = {}
            new_leaves.append(parent)
            new_tree.leaf = new_leaves
            cta = sum([n.eh for n in new_leaves]) + alpha * len(new_leaves)
            if cta <= ctb:
                # 判断剪枝完之后，新的子树是否是最佳树
                if cta < ctm:
                    best_tree = new_tree
                    ctm = cta
                # 对剪枝完的树继续进行剪枝
                sub_tree = self.cutting_tree_iteration(new_tree, alpha, best_tree, ctm)
                # 查看新生成的最佳树是不是当前最佳树
                ctbt = sum([n.eh for n in best_tree.leaf]) + alpha * len(best_tree.leaf)
                if ctbt < ctm:
                    best_tree = sub_tree
                    ctm = ctbt
        return best_tree


class ID_3(DecisionTree):

    def __init__(self, data: pd.DataFrame, target, threshold) -> None:
        super().__init__(data, target, threshold, information_gain)


class C45(DecisionTree):

    def __init__(self, data: pd.DataFrame, target, threshold) -> None:
        super().__init__(data, target, threshold, information_gain_rate)



if __name__ == '__main__':
    d = pd.read_csv("/Users/0g/Documents/2425s/learning/StatisticalLearningMethods/5.决策树/data1.csv", header=0, index_col=0)
    tree = ID_3(d, '类别', 0)
    tree.cutting_tree_iteration(tree.tree, 1)
    print(1)
            

            


    