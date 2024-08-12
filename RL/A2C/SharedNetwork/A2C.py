import torch

from A2C_Model import A2C_Model
import ARG

class A2C:
    def __init__(self, n_state, n_action, device):
        self.device = device
        self.model = A2C_Model(n_state, n_action).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), ARG.lr)
        self.mse = torch.nn.MSELoss()
        
    def selectAction(self, state):
        prob, _ = self.model(state)
        distribution = torch.distributions.Categorical(prob)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def save(self):
        torch.save(self.model.state_dict(), "saved/A2C.pth")
        
    def load(self, modelPath: str):
        loaded = torch.load(modelPath)
        self.model.load_state_dict(loaded)