import sklearn
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X_train, y_train = load_iris(return_X_y=True)
X = X_train[:, [0, 1]]
y = y_train
clf = Perceptron(fit_intercept=True,
                 max_iter=1000,
                 tol=None,
                 shuffle=True)

clf.fit(X, y)

# 画布大小
plt.figure(figsize=(10, 10))

# 中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(X[:50, 0], X[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(X[50:100, 0], X[50:100, 1], c='orange', label='Iris-versicolor')
#plt.scatter(X[100:150, 0], X[100:150, 1], c='red', label='Iris-virginica')
# 画感知机的线
x_ponits = np.arange(4, 8)
y1_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_[0])/clf.coef_[0][1]
# y2_ = -(clf.coef_[1][0]*x_ponits + clf.intercept_[1])/clf.coef_[1][1]
plt.plot(x_ponits, y1_)
# plt.plot(x_ponits, y2_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()