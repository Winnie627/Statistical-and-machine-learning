import pandas as pd
import numpy as np
lambda_ = 0  # e4.1
#lambda_ = 1  # e4.2
data = pd.read_csv("./data_4-1.txt", header=None, sep=",")
X = data[data.columns[0:2]]  # Series数据类型
y = data[data.columns[2]]
X = pd.DataFrame(X)  # 转为DataFrame数据类型
y = pd.DataFrame(y)
classes = np.unique(y)  # 类别标签
# >>[-1，1]
# https://www.cnblogs.com/xshan/p/10289588.html
# Series数据类型
class_count = y[y.columns[0]].value_counts()  # 计算每个类别的数量
# >>1 9; -1 6
class_prior = (class_count+lambda_)/(y.shape[0]+len(classes)*lambda_)  # 计算每一类的先验概率
# >>1 0.6; -1 0.4
prior = dict()  # 计算先验条件概率分布
# 外层循环体执行的次数 外层循环次数
# 内层循环体执行的次数 外层循环次数*内层循环次数
# idx=0 y=-1;idx=0 y=1;idx=1 y=-1;idx=-1 y=1
for idx in X.columns:  # 遍历X的列索引 >>[0,1] 对应训练数据的属性个数
    for j in classes:  # 遍历类别标签[-1,1]
        p_x_y = X[(y == j).values][idx].value_counts()  # 计算先验概率
        for i in p_x_y.index:  # [1, 2, 3]; [3, 2, 1]; ['S', 'M', 'L']; ['M', 'L', 'S']
            prior[(idx, i, j)] = (p_x_y[i]+lambda_)/(class_count[j]+len(np.unique(X[idx]))*lambda_)  # np.unique(X[idx]) idx属性对应的特征列表

target = [2, "S"]
rst = []
for class_ in classes:  # >>[-1,1]
    py = class_prior[class_]  # Series数据类型索引
    pxy = 1
    for idx, x in enumerate(target):  # >>[0,2] [1,"S"]
        pxy *= prior[(idx, x, class_)]
    rst.append(py*pxy)
print(rst)
result = classes[np.argmax(rst)]
print(result)