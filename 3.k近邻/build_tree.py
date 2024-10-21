import numpy as np
from nodes import Nodes


def kd_tree(T, j=0, pre=None):
    # 使用递归方法来构建kd树
    # 三个关键点：函数目标、退出条件、等价关系
    # 函数目标：构建kd树
    # 退出条件：当无节点时结束
    # 等价关系：输入数组的大小
    """
    采用中位数作为切分位点
    每次切分，都切换一个数据维度
    """
    k, n = T.shape
    # 退出条件
    if n == 1:
        tmp = Nodes(tuple(T.flatten()))
        tmp.pre = pre
        return tmp
    if n == 0:
        return None

    # 选择当前次切分的维度
    l = int(j % k)

    # 获取所选维度的所有数据
    tmp = np.roll(T, k-l-1, axis=0)
    T = T[:,np.lexsort(tmp)]
    l_dim = T[l]

    # 获取中位数
    median = np.median(l_dim)
    if median not in l_dim:
        median = l_dim[len(l_dim) // 2]

    this = tuple(T[:, T[l]==median].flatten())
    this = Nodes(this, l)
    this.pre = pre

    # 开始递归
    # 左子树
    left = kd_tree(T[:, T[l] < median], j+1, this)
    # 右子树
    right = kd_tree(T[:, T[l] > median], j+1, this)

    this.left = left
    this.right = right
    return this


if __name__ == '__main__':
    t = np.array([
        [2, 5, 9, 4, 8, 7],
        [3, 4, 6, 7, 1, 2]
    ])
    n = kd_tree(t)
    print(1)
