from ValueNN import ValueNN
from PolicyNN import PolicyNN
import torch

import ARG

class PPO:
    def __init__(self, stateDim, actionDim, device):
        self.device = device
        self.actor = PolicyNN(stateDim, actionDim).to(self.device)
        self.critic = ValueNN(stateDim).to(self.device)
        
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), ARG.Policy_lr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), ARG.Vaule_lr)
        
    def selectAction(self, state):
        prob = self.actor(state)
        distribution = torch.distributions.Categorical(prob)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def save(self):
        torch.save(self.actor.state_dict(), "actorModel.pth")
        torch.save(self.critic.state_dict(), "criticModel.pth")
        
    def load(self, actorModelPath: str, criticModelPath: str):
        loaded = torch.load(actorModelPath)
        self.actor.load_state_dict(loaded)
        loaded = torch.load(criticModelPath)
        self.critic.load_state_dict(loaded)