from Env import Env
from DDPG import DDPG
import torch as th
import numpy as np
import random

def setSeed(seed = 42, loader = None):
    random.seed(seed) 
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed) 
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True        
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

if __name__ == "__main__":
    setSeed()
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    env = Env(device, "Pendulum-v1", "rgb_array")
    agent = DDPG(env.n_state, env.n_action, 2, device)
    env.train(agent, 100000)
    agent.save()