import numpy as np
import typing
import Tensor
import abc

class Loss:
    def __init__(self, loss, delta):
        self.loss = loss
        self.delta = delta

class LossFunc:
    def __init__(self):
        pass
    @abc.abstractmethod
    def apply(self, pred: typing.Union[Tensor.Tensor, np.ndarray], target: typing.Union[Tensor.Tensor, np.ndarray]) -> Loss:
        pass
    def __call__(self, pred: typing.Union[Tensor.Tensor, np.ndarray], target: typing.Union[Tensor.Tensor, np.ndarray]) -> Loss:
        return self.apply(pred, target)

class MSE(LossFunc):
    def __init__(self):
        super().__init__()
    def apply(self, pred: typing.Union[Tensor.Tensor, np.ndarray], target: typing.Union[Tensor.Tensor, np.ndarray]) -> Loss:
        if (isinstance(pred, Tensor.Tensor)):
            pred = pred._data
        if (isinstance(target, Tensor.Tensor)):
            target = target._data
        loss = np.mean(np.square(pred - target))
        return Loss(loss, (pred - target) / target.shape[0])