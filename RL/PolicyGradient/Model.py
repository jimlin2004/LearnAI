import torch

class Model(torch.nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.l1 = torch.nn.Linear(inDim, 8)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(8, outDim)
        self.allActionProb = torch.nn.Softmax(dim = 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.allActionProb(out)
        return out