import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._order = 0
        self.error = None
    @property
    def shape(self):
        return self._data.shape
    def numpy(self):
        return self._data
    def setError(self, error):
        self.error = error