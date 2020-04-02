import numpy as np
import random


class Model:

    def __init__(self):

        self.eta = 0.1

    def sign(self, w, x):
        y = np.dot(w, x)
        return y

    def fit(self, X_train, y_train):
        self.w = np.zeros(X_train.shape[1]+1)
        max_iter = 1000
        n = 0
        correct_count = 0
        while n < max_iter:
            index = random.randint(0, y_train.shape[0] - 1)
            X = np.hstack([X_train[index], 1]).T
            y = 2 * y_train[index] - 1
            if self.sign(self.w, X) * y > 0:
                correct_count += 1
                if correct_count > max_iter:
                    break
                continue
            self.w += y * X
            n += 1
            if True:
                print("n", n)
