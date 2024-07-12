# 定義NN Layer

import typing
import abc
import numpy as np
import Tensor

class BaseLayer:
    def __init__(self):
        self.inTensorPtr = None
        self.outTensorPtr = None
        
        # 用於記住ptr實際的記憶體
        self.out = None
        self._order = 0
        self.name = None
    def _processInput(self, x: typing.Union[Tensor.Tensor, np.ndarray]) -> Tensor.Tensor:
        if (isinstance(x, np.ndarray)):
            x = Tensor.Tensor(x)
            x._order = 0
        self.inTensorPtr = x
        return x
    def _processOutput(self, x: np.ndarray, order: int) -> Tensor.Tensor:
        self.outTensorPtr = Tensor.Tensor(x)
        self.outTensorPtr._order = order
        return self.outTensorPtr
    @abc.abstractmethod
    def forward(self, x: typing.Union[Tensor.Tensor, np.ndarray]) -> Tensor.Tensor:
        pass
    @abc.abstractmethod
    def backward(self):
        pass
    def __call__(self, x: typing.Union[Tensor.Tensor, np.ndarray]) -> Tensor.Tensor:
        return self.forward(x)

class BaseActivationFunc(BaseLayer):
    def __init__(self):
        pass

class ReLU(BaseActivationFunc):
    def __init__(self):
        super().__init__()
    def forward(self, x: typing.Union[Tensor.Tensor, np.ndarray]):
        x = self._processInput(x)
        self._order = x._order
        _x = np.maximum(0, x._data)
        self.out = self._processOutput(_x, self._order + 1)
        # print(f"IN: {self.inTensorPtr._data}")
        # print(f"OUT: {self.outTensorPtr._data}")
        return self.out
    def backward(self):
        dz = self.outTensorPtr.error
        newdz = dz * self.derivative(self.inTensorPtr._data)
        # print(newdz)
        self.inTensorPtr.setError(newdz)
        return None
        
    def derivative(self, x: np.ndarray):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

class Linear(BaseLayer):
    def __init__(self, inDim: int, outDim: int):
        super().__init__()
        self.weight = np.random.rand(inDim, outDim)
        self.bias = np.random.rand(1, outDim)
    def forward(self, x: typing.Union[Tensor.Tensor, np.ndarray]) -> Tensor.Tensor:
        x = self._processInput(x)
        self._order = x._order
        # print(f"forward: {self._order}")
        _x = np.dot(x._data, self.weight) + self.bias
        self.out = self._processOutput(_x, self._order + 1)
        
        # print(f"IN: {self.inTensorPtr._data}")
        # print(f"OUT: {self.outTensorPtr._data}")
        return self.out
    def backward(self):
        # 計算gw, gb
        dz = self.outTensorPtr.error
        gradients = {}
        gradients["gw"] = np.dot(self.inTensorPtr._data.T, dz)
        gradients["gb"] = np.sum(dz, axis = 0, keepdims = True)
        self.inTensorPtr.setError(np.dot(dz, self.weight.T))
        
        self.weight -= 0.01 * gradients["gw"]
        self.bias -= 0.01 * gradients["gb"]
        return gradients