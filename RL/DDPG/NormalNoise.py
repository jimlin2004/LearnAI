import numpy as np

class NormalNoise:
    def __init__(self, mean, sigma):
        self.mu = mean
        self.sigma = sigma
    def __call__(self):
        return np.random.normal(self.mu, self.sigma)