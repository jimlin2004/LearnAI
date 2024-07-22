from Env import Env
from PPO import PPO
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = Env(device, "CartPole-v1", "rgb_array")
    agent = PPO(env.stateDim, env.actionDim, device)
    agent.load("loadModel/actorModel.pth", "loadModel/criticModel.pth")
    env.evaluate(agent)