from Env import Env
from DDPG import DDPG
import torch as th
import numpy as np
import random

if __name__ == "__main__":
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    env = Env(device, "Pendulum-v1", "rgb_array")
    agent = DDPG(env.n_state, env.n_action, 2, device)
    agent.load("./loadModel/Actor.pth", "./loadModel/Critic.pth")
    env.evaluate(agent, 5)