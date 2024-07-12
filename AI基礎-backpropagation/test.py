import numpy as np

a = np.array([1])
b = np.random.rand(1, 4)
print(np.dot(a, b))
print(a.dot(b))
print(b.dot(a))