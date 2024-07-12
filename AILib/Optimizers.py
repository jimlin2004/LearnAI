import numpy as np
import abc

class Optimizer:
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
    @abc.abstractmethod
    def step(self):
        pass
    
class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
    def step(self):
        print(self._params)