from CriticNN import CriticNN
from ActorNN import ActorNN
import torch

import ARG

class PPO:
    def __init__(self, stateDim, bound, device):
        self.device = device
        self.bound = bound
        self.actor = ActorNN(stateDim, bound).to(self.device)
        self.critic = CriticNN(stateDim).to(self.device)
        
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), ARG.Actor_lr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), ARG.Critic_lr)
        
    def selectAction(self, state):
        mu, sigma = self.actor(state)
        distribution = torch.distributions.Normal(mu, sigma)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        # 不能超過遊戲的限制範圍
        action.clamp(-self.bound, self.bound)
        return action.item(), log_prob
    
    def save(self):
        torch.save(self.actor.state_dict(), "actorModel.pth")
        torch.save(self.critic.state_dict(), "criticModel.pth")
        
    def load(self, actorModelPath: str, criticModelPath: str):
        loaded = torch.load(actorModelPath)
        self.actor.load_state_dict(loaded)
        loaded = torch.load(criticModelPath)
        self.critic.load_state_dict(loaded)