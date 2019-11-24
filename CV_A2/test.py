import numpy as np
a = np.array([[1, 2, 3], [3, 6, 9], [1, 2, 3]])
b = np.array([1, 2, 3])
c = np.array([1/3, 1/3, 1/3])

# b = b[:, None]
print(np.sum(a[:, 0]))
print(np.matmul(a, b))
print(np.matmul(a, b[:, None]))
print(np.matmul(b, a))
print(np.dot(b, c))
print(np.ones((3, 3))/9)