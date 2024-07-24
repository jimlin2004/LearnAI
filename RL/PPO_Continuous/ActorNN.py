import torch

class ActorNN(torch.nn.Module):
    def __init__(self, inDim, bound):
        super().__init__()
        self.bound = bound
        
        self.l1 = torch.nn.Linear(inDim, 32)
        self.relu1 = torch.nn.ReLU()
        self.mu_out = torch.nn.Linear(32, 1)
        self.tanh1 = torch.nn.Tanh()
        self.sigma_out = torch.nn.Linear(32, 1)
        self.softplus1 = torch.nn.Softplus()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        mu = self.mu_out(out)
        mu = self.bound * self.tanh1(mu)
        # sigma為正，所以要套一個類似relu的activation
        sigma = self.sigma_out(out)
        sigma = self.softplus1(sigma)
        return mu, sigma