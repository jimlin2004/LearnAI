import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from DQN import DQN
from Logger import Logger
import ARG

def saveFramesToGif(frames, gifPath):
    patch = plt.imshow(frames[0])
    plt.axis("off")
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames))
    anim.save(gifPath, writer = "ffmpeg", fps = 30)

class Env:
    def __init__(self, device, gameName: str, renderMode = "human"):
        self.env = gym.make(gameName, render_mode = renderMode)
        self.history = {
            "reward": []
        }
        self.device = device
        self.logger = Logger()
    
    def runBatchEpisode(self, agent: DQN):
        t = 0
        while (t < ARG.Max_Timesteps_Per_Batch):
            state = self.env.reset()[0]
            ep_reward = 0
            ep_loss = 0
            ep_t = 1
            for ep_t in range(1, ARG.Max_Timesteps_Per_Episode + 1):
                t += 1
                stateTensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
                action = agent.selectAction(stateTensor)
                nextState, reward, done, _, _ = self.env.step(action)
                # 這邊計算用原本的reward function
                ep_reward += reward
                # 改成自訂的reward function
                x, v, theta, omega = nextState
                # 車離x軸中間越近越好
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold
                # 火柴角度越小越好
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians
                reward = r1 + r2
                agent.storeTransition(state, action, nextState, reward, done)
                if (done):
                    break
                if (len(agent.replayBuffer) >= ARG.BatchSize):
                    loss = agent.train()
                    ep_loss += loss
                state = nextState
                if (ep_t == ARG.Max_Timesteps_Per_Episode):
                    break
                if (t >= ARG.Max_Timesteps_Per_Batch):
                    break
            self.logger.history["reward"].append(ep_reward)
            self.logger.history["loss"].append(ep_loss / ep_t)
            print("| Reward: %5.6f | Loss: %5.6f | Epsilon: %5.6f |" % (ep_reward, ep_loss / ep_t, agent.epsilon))
            agent.updateEpsilon()
        return t
            
    def evaluate(self, agent: DQN):
        frames = []
        currTimesteps = 0
        state = self.env.reset()[0]
        while (currTimesteps < 1000):
            currTimesteps += 1
            frames.append(self.env.render())
            state = torch.tensor(state, dtype = torch.float).unsqueeze(0).to(self.device)
            action, _ = agent.selectAction(state)
            nextState, reward, done, _, _ = self.env.step(action)
            if (done):
                break
            state = nextState
        self.env.close()
        saveFramesToGif(frames, "./evaluate.gif")
        
    def train(self, agent: DQN, totalTimesteps: int):
        currTimesteps = 0
        while (currTimesteps < totalTimesteps):
            currTimesteps += self.runBatchEpisode(agent)
        self.env.close()
        self.logger.saveHistory()
    @property
    def actionDim(self):
        return self.env.action_space.n
    @property
    def stateDim(self):
        return self.env.observation_space.shape[0]