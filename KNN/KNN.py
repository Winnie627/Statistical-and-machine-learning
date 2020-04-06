# e3.2 Kd平衡树 + q3.2 k近邻搜索
import numpy as np
from collections import namedtuple
# 定义递归函数


# 生成kd树
def fix(X, depth=0):
    print(X)
    k = X.shape[1]
    axis = depth % k
    X = X[X[:, axis].argsort()]
    median = X.shape[0] // 2  # 从小到大的数组值索引
    try:
        X[median]
    except IndexError:
        return None
    location = X[median]
    left_child = fix(X[:median], depth+1)
    right_child = fix(X[median+1:], depth+1)
    Node = namedtuple('Node', 'location left_child right_child')
    return Node(location, left_child, right_child)


result = namedtuple("Result_tuple",
                    "nearest_point nearest_dist nodes_visited")  # 存储最近点 最近距离 访问过的节点数


def search(point, tree, max_dist, depth=0):
    k = len(point)  # 目标点维度
    if tree is None:
        return result([0]*k, float("inf"), 0)

    nodes_visited = 1
    s = depth % k  # 进行分割的参考维度序号

    if point[s] < tree.location[s]:
        nearer_node = tree.left_child
        further_node = tree.right_child
    else:
        nearer_node = tree.right_child
        further_node = tree.left_child

    temp1 = search(point, nearer_node, max_dist, depth=depth+1)  # 遍历找到目标实例点的区域

    nearest = temp1.nearest_point  # 以此叶节点为 当前最近点
    dist = temp1.nearest_dist  # 更新最近点与目标实例点的距离

    nodes_visited += temp1.nodes_visited

    if dist < max_dist:
        max_dist = dist  # 最近距离点在以目标实例点为球心，max_dist为半径的超球体内

    temp_dist = np.linalg.norm(point-tree.location, ord=2)  # 目标实例点和分割点的欧氏距离
    if temp_dist > max_dist:  # 如果没有分割点在超球体内，则直接返回叶节点为最近点，得到结果
        return result(nearest, dist, nodes_visited)

    if temp_dist < dist:
        nearest = tree.location
        dist = temp_dist  # 更新最近点与目标实例点的距离
        max_dist = dist  # 更新超球体半径

    temp2 = search(point, further_node, max_dist, depth=depth+1)  # 同时检查更远节点区域
    nodes_visited += temp2.nodes_visited
    if temp2.nearest_dist < dist:
        nearest = temp2.nearest_point  # 更新最近点
        dist = temp2.nearest_dist  # 更新最近距离

    return result(nearest, dist, nodes_visited)


X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
tree = fix(X)
target = np.array([3, 4.5])
ret = search(target, tree, float("inf"), 0)
print(tree)
print(ret)
