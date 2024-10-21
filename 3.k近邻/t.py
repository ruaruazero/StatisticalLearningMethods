import numpy as np

# 使用递归方法来构建kd树
# 三个关键点：函数目标、退出条件、等价关系
# 函数目标：构建kd树
# 退出条件：当无节点时结束
# 等价关系：输入数组的大小
"""
采用中位数作为切分位点
每次切分，都切换一个数据维度
"""
def kd_tree(nodes, j=0):

    k, n = nodes.shape
    # 退出条件
    if n == 1:
        return tuple(nodes.flatten())  
    if n == 0:
        return None

    # 选择当前次切分的维度
    l = int(j % k)

    # 获取所选维度的所有数据
    tmp = np.roll(nodes, k-l-1, axis=0)
    nodes = nodes[:,np.lexsort(tmp)]
    l_dim = nodes[l]

    # 获取中位数
    median = np.median(l_dim)
    if median not in l_dim:
        median = l_dim[len(l_dim) // 2]

    this = list(np.append(nodes[:, nodes[l]==median].flatten(), (l, median)))
    this = tuple([int(i) for i in this])

    # 开始递归
    # 左子树
    left = kd_tree(nodes[:, nodes[l] < median], j+1)
    # 右子树
    right = kd_tree(nodes[:, nodes[l] > median], j+1)

    tree = [this, [left, right]]
    return tree


def find_leaf(tree, point):
    # 退出条件，当前节点为叶节点
    if isinstance(tree, tuple):
        return ['$']
    
    # 查找左右子树
    k = tree[0][2]
    thre = tree[0][3]
    if point[k] < thre:
        if tree[1][0] is None:
            return ['0$']
        path = ['0'] + find_leaf(tree[1][0], point)
    elif point[k] > thre:
        if tree[1][1] is None:
            return ['1$']
        path = ['1'] + find_leaf(tree[1][1], point)
    else:
        return ['$']
    return path


if __name__ == '__main__':
    T = np.array([
        [2, 5, 9, 4, 8, 7],
        [3, 4, 6, 7, 1, 2]
    ])
    tre = kd_tree(T)
    print(find_leaf(tre, (9, 7)))
