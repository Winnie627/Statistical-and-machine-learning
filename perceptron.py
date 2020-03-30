# e2.1
import numpy as np
import random
x1 = np.array([3, 3])
y1 = 1
x2 = np.array([4, 3])
y2 = 1
x3 = np.array([1, 1])
y3 = -1
X = np.array([x1, x2, x3])
Y = np.array([y1, y2, y3])
# data_raw = np.loadtxt("D:/Pycharmproject/Lihang-master/CH02/Input/data_2-1.txt")
# X = data_raw[:, :2]
# y = data_raw[:, -1]
# print(X, y)

max_iter = 100
n = 0
w = np.zeros(X.shape[1]+1)  # add b to vector w
correct_count = 0

# 随机梯度下降法迭代
while n < max_iter:
    index = random.randint(0, Y.shape[0]-1)
    x = np.hstack([X[index], 1])
    y = 2*Y[index] - 1
    wx = np.dot(w, x)
    if wx*y > 0:
        correct_count += 1
        if correct_count > max_iter:
            break
        continue
    w += y*x
    n += 1
    if True:
        print(n)
print(w)  # 选取出模型参数w,b

# 给定一组数据，即输入仍为X，输出均为1，判断数据点的ture/false
X = np.hstack([X, np.ones(X.shape[0]).reshape((-1, 1))])

# predict the given 3 group data,1 for ture, -1 for false
rst = np.array([1 if rst else -1 for rst in np.dot(X, w) > 0])
# >>[1 1 -1]

