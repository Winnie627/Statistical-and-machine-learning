import math
import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.model = None
        self.classes = None
        self.class_prior = None
        self.class_count = None

    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    def std(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    def guassian_probability(self, x, mean, std):
        exponent = math.exp(-(math.pow(x - mean, 2)/(2*math.pow(std, 2))))
        return (1/(math.sqrt(2*math.pi)*std))*exponent

    def summarize(self, train_data):
        summarizes = [(self.mean(i), self.std(i)) for i in zip(*train_data)]
        return summarizes

    def fit(self, X, y):
        self.classes = np.unique(y)
        y_ = pd.DataFrame(y)
        self.class_count = y_[y_.columns[0]].value_counts()
        self.class_prior = self.class_count / y_.shape[0]
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        # model包含四个属性，每个属性在label0和label1下的均值和方差
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'

    def calculate_probabilities(self, input_data):
        probabilities = {}
        rst = {}
        for class_ in self.classes:
            py = self.class_prior[class_]
            for label, value in self.model.items():
                probabilities[label] = 1
                for i in range(len(value)):
                    mean, std = value[i]
                    probabilities[label] *= self.guassian_probability(
                        input_data[i], mean, std
                    )
                rst[label] = py*probabilities[label]
        return rst

    def predict(self, X_test):  # 后验概率
        label = sorted(  # 对每一组测试数据，计算概率最高的类别，输出label
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        print(sorted(  # 对每一组测试数据，计算概率最高的类别，输出label
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0])
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right +=1
        return right / float(len(X_test))


