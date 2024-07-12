import Layer
import Loss
import numpy as np
import abc

class Module:
    def __init__(self):
        self.params = {}
        self.sortedLayers = []
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    def backward(self, loss: Loss.Loss):
        if (len(self.sortedLayers) == 0):
            allLayer = []
            for name, var in self.__dict__.items():
                if (isinstance(var, Layer.BaseLayer)):
                    var.name = name
                    allLayer.append(var)
                elif (isinstance(var, Layer.BaseActivationFunc)):
                    var.name = name
                    allLayer.append(var)
            allLayer.sort(key = lambda item: item._order)
            self.sortedLayers = allLayer
        
        lastLayer = self.sortedLayers[-1]
        lastLayer.outTensorPtr.setError(loss.delta)
        for layer in self.sortedLayers[::-1]:
            gradients = layer.backward()
            # print(gradients)
            if (isinstance(layer, Layer.BaseActivationFunc)):
                continue
            self.params[layer.name] = {"gradient": {}}
            for key in gradients.keys():
                self.params[layer.name]["gradient"][key] = gradients[key]

    def __call__(self, x: np.ndarray) ->np.ndarray:
        return self.forward(x)
    def parameters(self):
        return self.params