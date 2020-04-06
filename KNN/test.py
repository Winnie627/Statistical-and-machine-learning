from sklearn.datasets import load_iris
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, X_train, y_train,n_neighbors=3,p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):  # 遍历所有数据点，找到距离测试实例点最近的n个点里多数是什么类别，该测试实例点就是什么类别
        knn_list = []
        # 取n个点
        for i in range(self.n):
            dist = np.linalg.norm(X-self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))  # 构建二维列表，每个元素由距离，类别构成
        # 保证选取距离最近的n个点
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))  # 选取最大距离（0维数据）对应的索引
            dist = np.linalg.norm(X-self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:  # 更新最大距离数据点
                knn_list[max_index] = (dist, self.y_train[i])
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)  # 返回以元素为key，元素个数为value的对象集合
        # 字典items：返回可遍历的键 值, 按值大小排序
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]  # 最大值对应的键作为max_count
        return max_count  # 选择类别数量多的作为预测结果

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


X_, y_ = load_iris(return_X_y=True)
X = X_[:100, [0, 1]]
y = y_[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = KNN(X_train, y_train)
point = np.array([6, 3])
clf_pre = clf.predict(point)
print(clf_pre)
clf_s = clf.score(X_test, y_test)
print(clf_s)

plt.scatter(X[:50, 0], X[:50, 1], c='b', label='0',)
plt.scatter(X[50:100, 0], X[50:100, 1], c='orange', label='1')
plt.scatter(point[0], point[1], c='red', label='test')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()