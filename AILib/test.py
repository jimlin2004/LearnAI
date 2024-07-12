import Layer
import Loss
import Module
import Optimizers
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class Model(Module.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Layer.Linear(1, 10)
        self.relu1 = Layer.ReLU()
        self.l2 = Layer.Linear(10, 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        return out

if __name__ == "__main__":
    X, Y = make_regression(100, n_features = 1, n_targets = 1, noise = 10, random_state = 3)
    Y = Y.reshape(100, 1)
    model = Model()
    optimizer = Optimizers.SGD(model.parameters(), 0.01)
    lossFunc = Loss.MSE()
    for i in range(50):
        prediction = model(X).numpy()
        loss = lossFunc(prediction, Y)
        print(loss.loss)
        model.backward(loss)
        # optimizer.step()

    prediction = model(X).numpy()
    plt.scatter(X, Y)
    plt.scatter(X, prediction)
    plt.show()