import torch

class A2C_Model(torch.nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.l1 = torch.nn.Linear(n_state, 32)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        
        self.actorOut = torch.nn.Linear(32, n_action)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.criticOut = torch.nn.Linear(32, 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        
        action_prob = self.softmax(self.actorOut(out))
        V = self.criticOut(out)
        return action_prob, V