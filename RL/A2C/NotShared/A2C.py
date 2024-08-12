import torch

from PolicyNN import PolicyNN
from ValueNN import ValueNN
import ARG

class A2C:
    def __init__(self, n_state, n_action, device):
        self.device = device
        self.actor = PolicyNN(n_state, n_action).to(device)
        self.critic = ValueNN(n_state).to(device)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), ARG.actor_lr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), ARG.critic_lr)
        self.mse = torch.nn.MSELoss()
        
    def selectAction(self, state):
        prob = self.actor(state)
        distribution = torch.distributions.Categorical(prob)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def save(self):
        torch.save(self.actor.state_dict(), "saved/actor.pth")
        torch.save(self.critic.state_dict(), "saved/critic.pth")
        
    def load(self, actorModelPath: str, criticModelPath: str):
        loaded = torch.load(actorModelPath)
        self.actor.load_state_dict(loaded)
        loaded = torch.load(criticModelPath)
        self.critic.load_state_dict(loaded)