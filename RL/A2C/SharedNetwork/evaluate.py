from Env import Env
from A2C import A2C
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = Env(device, "CartPole-v1", "rgb_array")
    agent = A2C(env.n_state, env.n_action, device)
    agent.load("./loadModel/A2C.pth")
    env.evaluate(agent)