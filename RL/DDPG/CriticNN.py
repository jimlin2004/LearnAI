import torch as th

class CriticNN(th.nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.l1 = th.nn.Linear(n_state + n_action, 32)
        self.relu1 = th.nn.ReLU()
        self.l2 = th.nn.Linear(32, 32)
        self.relu2 = th.nn.ReLU()
        self.l3 = th.nn.Linear(32, 1)
    def forward(self, state, action):
        x = th.cat([state, action], dim = 1)
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out