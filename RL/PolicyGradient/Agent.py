import Model
import torch
import numpy as np

class Agent:
    def __init__(self, device, stateDim, actionDim):
        self.lr = 0.01
        self.gamma = 0.99
        self.log_probs = []
        self.ep_rewards = []
        self.policy = Model.Model(stateDim, actionDim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)
        self.device = device
    
    def selectAction(self, state):
        actionProb = self.policy(state)
        distri = torch.distributions.Categorical(actionProb)
        action = distri.sample()
        log_prob = distri.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()
    
    def getDiscountedAndStandardizedRewards(self):
        discountedRewards = [0] * len(self.ep_rewards)
        curr = 0
        for i in range(len(self.ep_rewards) - 1, -1, -1):
            curr = curr * self.gamma + self.ep_rewards[i]
            discountedRewards[i] = curr
        discountedRewards = torch.FloatTensor(discountedRewards)
        discountedRewards = (discountedRewards - discountedRewards.mean()) / (discountedRewards.std() + 1e-9)
        return discountedRewards
    
    def train(self, discountedAndStandardizedRewards: torch.Tensor):
        # self.policy.train(True)
        policyLoss = []
        for log_prob, R in zip(self.log_probs, discountedAndStandardizedRewards):
            policyLoss.append(-log_prob * R)
        policyLoss = torch.cat(policyLoss).sum().to(self.device)
        self.optimizer.zero_grad()
        policyLoss.backward()
        self.optimizer.step()
        return policyLoss.item()
    
    def save(self, path: str):
        torch.save(self.policy.state_dict, path)