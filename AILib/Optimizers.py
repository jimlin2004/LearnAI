import numpy as np
import abc

class Optimizer:
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
    @abc.abstractmethod
    def step(self):
        pass
    
class GD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
    def step(self):
        for layername, mp in self._params.items():
            gradient = mp["gradient"]
            layer = mp["layer"]
            layer.weight -= self._lr * gradient["gw"]
            layer.bias -= self._lr * gradient["gb"]