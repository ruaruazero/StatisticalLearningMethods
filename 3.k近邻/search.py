from nodes import Nodes
import numpy as np
from build_tree import kd_tree


def find_leaf(tree: Nodes, point, p=2):
    # 退出条件，当前节点为叶节点
    if tree.left is None and tree.right is None:
        return tree
    
    # 查找左右子树
    k = tree.dim
    thre = tree.data[k]
    if point[k] < thre:
        if tree.left is None:
            return tree
        tree = find_leaf(tree.left, point)
    elif point[k] > thre:
        if tree.right is None:
            return tree
        tree = find_leaf(tree.right, point)
    else:
        if tree.left is not None and tree.right is not None:
            if distance(tree.left, point, p) < distance(tree.right, point, p):
                tree = find_leaf(tree.left, point)
            else:
                tree = find_leaf(tree.right, point)
        else:
            if tree.right is not None:
                tree = find_leaf(tree.right, point)
            else:
                tree = find_leaf(tree.left, point)
    return tree


def distance(node: Nodes, point, p=2):
    node_data = np.array(node.data)
    point_data = np.array(point)
    return (np.abs(node_data - point_data) ** p).sum() ** (1 / p)


def recursive_search(node: Nodes, point, nearest=None, p=2, another_region=None):
    """
    结束条件: 1.回退到根节点 2.当前节点的另一子节点的区域是否相交
    node: 当前查看的节点
    point: 目标节点
    nearest: 最近节点
    """
    # 如果是首层递归, 那么最近邻就是近似最近邻
    if nearest is None:
        nearest = node
    # 已回到根节点, 递归结束
    if node is None:
        return nearest
    if node.pre is None:
        return nearest
    # 递归查找父节点
    tmp = node.pre
    if distance(tmp, point, p) <= distance(nearest, point, p):
        # 针对父节点，如果父节点的距离更近，将最近点转移到当前点
        nearest = tmp
    k = tmp.dim
    thre = tmp.data[k]
    is_reached = np.abs(point[k] - thre) < distance(nearest, point, p)
    # 如果和另一子树不存在交集
    if not is_reached:
        nearest = recursive_search(tmp.pre, point, nearest, p)
    # 如果和另一子树存在交集
    else:
        if tmp.left == node:
            # 查看右子树
            if another_region == tmp.right:
                return nearest
            n = find_leaf(tmp.right, point)
            nearest = recursive_search(n, point, p=p, another_region=tmp.left)      
        else:
            # 查看左子树
            if another_region == tmp.left:
                return nearest
            n = find_leaf(tmp.left, point, tmp.right)
            nearest = recursive_search(n, point, p=p, another_region=tmp.right)
    return nearest


def nearest_search(tree, point):
    # 找到当前点在kd树中的叶节点
    nearest = find_leaf(tree, point)
    # 判断是否为叶节点
    is_leaf = nearest.left is None and nearest.right is None
    # 若非叶节点，搜索当前节点的子节点
    if not is_leaf:
        if nearest.left is not None:
            nearest = nearest.left
        else:
            nearest = nearest.right
    nearest = recursive_search(nearest, point)
    return nearest


if __name__ == '__main__':
    t = np.array([
        [2, 5, 9, 4, 8, 7],
        [3, 4, 6, 7, 1, 2]
    ])
    n = kd_tree(t)
    n = nearest_search(n, (7, 6))
    print(1)