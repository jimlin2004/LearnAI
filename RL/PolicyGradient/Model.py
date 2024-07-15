import torch

class Model(torch.nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.l1 = torch.nn.Linear(inDim, 32)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(32, outDim)
        self.allActionProb = torch.nn.Softmax(dim = -1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.allActionProb(out)
        return out