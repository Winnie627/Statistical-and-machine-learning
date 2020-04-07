import pandas as pd
import numpy as np
lambda_ = 0  # e4.1
#lambda_ = 1  # e4.2
data = pd.read_csv("./data_4-1.txt", header=None, sep=",")
X = data[data.columns[0:2]]  # Series数据类型
y = data[data.columns[2]]
X = pd.DataFrame(X)  # 转为DataFrame数据类型
y = pd.DataFrame(y)
classes = np.unique(y)
class_count = y[y.columns[0]].value_counts()  # 计算每个类别的数量
# >>1 9; -1 6
class_prior = class_count+lambda_/y.shape[0]+len(classes)*lambda_  # 计算先验概率
# >>1 0.6; -1 0.4
prior = dict()
for idx in X.columns:
    for j in classes:
        p_x_y = X[(y == j).values][idx].value_counts()  # 计算先验概率
        for i in p_x_y.index:
            prior[(idx, i, j)] = p_x_y[i]+lambda_/class_count[j]+len(np.unique(X[idx]))*lambda_

target = [2, "S"]
rst = []
for class_ in classes:
    py = class_prior
    pxy = 1
    for idx, x in enumerate(target):
        pxy *= prior[(idx, x, class_)]
    rst.append(py*pxy)
print(rst)
result = classes[np.argmax(rst)]
print(result)