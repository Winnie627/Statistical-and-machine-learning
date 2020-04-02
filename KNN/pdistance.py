# e3.1
import numpy as np

X = np.array([[1, 1], [5, 1], [4, 4]])
for i in range(1, 5):
    r = {'1-{}'.format(c): np.linalg.norm(c-X[0], ord=i) for c in X[[1, 2], :]}  # 取1，2，3，4范数
    print(r)
    print(min(zip(r.values(), r.keys())))
