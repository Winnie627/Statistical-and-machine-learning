import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
X_, y_ = load_iris(return_X_y=True)
X = X_[:100, [0, 1]]
y = y_[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf_sk = KNeighborsClassifier()
print(clf_sk.fit(X_train, y_train))
print(clf_sk.score(X_test, y_test))
test_rst = clf_sk.predict(X_test)
print(clf_sk.predict(X_test))
print(y_test)
# 画布大小
plt.figure(figsize=(10, 10))

# 中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花k近邻数据示例')

plt.scatter(X[:50, 0], X[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(X[50:100, 0], X[50:100, 1], c='orange', label='Iris-versicolor')
for i in range(len(test_rst)):
    if test_rst[i] == y_test[i]:
        color = 'black'
    else:
        color = 'r'
plt.scatter(X_test[:, 0], X_test[:, 1], c=color, label='test_rst')

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()