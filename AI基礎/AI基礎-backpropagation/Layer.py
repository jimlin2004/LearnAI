import numpy as np

class Linear():
    def __init__(self, inDim: int, outDim: int):
        self.weight = np.random.uniform(-1, 1, size = [inDim, outDim])
        self.bias = np.random.uniform(-1, 1, size = [1, outDim])
    def __call__(self, x: np.ndarray):
        out = x.dot(self.weight) + self.bias
        return out

class Relu():
    def __init(self):
        pass
    def __call__(self, x: np.ndarray):
        return np.maximum(0, x)
    def derivative(self, x: np.ndarray):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))