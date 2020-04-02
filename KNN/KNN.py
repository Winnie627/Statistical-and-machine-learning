# e3.2 Kd平衡树
import numpy as np
from pprint import pformat
from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
# 定义迭代函数


def fix(X, depth=0):
    k = X.shape[1]
    axis = depth % k
    X = X[X[:, axis].argsort()]
    median = X.shape[0] // 2
    try:
        X[median]
    except IndexError:
        return None
    location = X[median]
    left_child = fix(X[:median], depth+1)
    right_child = fix(X[median+1:], depth+1)
    Node = namedtuple('Node', 'location left_child right_child')
    rst = Node(location, left_child, right_child)
    return pformat(tuple(rst))


X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
R = fix(X)
print(R)

