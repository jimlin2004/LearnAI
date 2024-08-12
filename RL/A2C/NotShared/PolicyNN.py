import torch

class PolicyNN(torch.nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.l1 = torch.nn.Linear(n_state, 32)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(32, n_action)
        self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.softmax(out)
        return out