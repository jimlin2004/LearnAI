import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 10)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(10, 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        return out

if __name__ == "__main__":
    X, Y = make_regression(100, n_features = 1, n_targets = 1, noise = 10, random_state = 3)
    Y = Y.reshape(100, 1)
    
    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))
    
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    lossFunc = torch.nn.MSELoss()
    for i in range(50):
        prediction = model(X)
        loss = lossFunc(prediction, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    prediction = model(X).detach().numpy()
    plt.scatter(X, prediction)
    plt.scatter(X, Y)
    plt.show()