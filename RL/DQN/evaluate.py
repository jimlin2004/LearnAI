from Env import Env
from DQN import DQN
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = Env(device, "CartPole-v1", "rgb_array")
    agent = DQN(env.stateDim, env.actionDim, device)
    agent.load("loadModel/DQN.pth")
    env.evaluate(agent)