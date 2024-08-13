import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from DDPG import DDPG
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
        self.device = device
        self.logger = Logger()
        self.trainTimestep = 0
    
    def runOneEpisode(self, agent: DDPG):
        state = self.env.reset()[0]
        ep_return = 0
        ep_t = 1
        ep_train_timestep = 0
        ep_actorLoss = 0
        ep_criticLoss = 0
        for ep_t in range(1, ARG.Max_Timesteps_Per_Episode + 1):
            stateTensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
            action = agent.selectAction(stateTensor)
            nextState, reward, done, _, _ = self.env.step(action)
            ep_return += reward
            agent.storeTransition(state, action, nextState, reward, done)
            
            if (len(agent.replayBuffer) > ARG.initReplayBufferSize):
                ep_train_timestep += 1
                actorLoss, criticLoss = agent.train()
                ep_actorLoss += actorLoss
                ep_criticLoss += criticLoss
            state = nextState
            
            if (done):
                break
            if (ep_t == ARG.Max_Timesteps_Per_Episode):
                break
        if (ep_train_timestep != 0):
            self.trainTimestep += ep_train_timestep
            self.logger.history["reward"].append(ep_return)
            self.logger.history["timestep"].append(self.trainTimestep)
            self.logger.history["actor loss"].append(ep_actorLoss / ep_train_timestep)
            self.logger.history["critic loss"].append(ep_criticLoss / ep_train_timestep)
            print("| Episode reward: %10.4f | Actor loss: %10.4f | Critic loss: %10.4f | normal sigma: %6f |" % (ep_return, ep_actorLoss / ep_train_timestep, ep_criticLoss / ep_train_timestep, agent.normalNoise.sigma))
        return ep_train_timestep

    def train(self, agent: DDPG, totalTimesteps: int):
        currTimesteps = 0
        while (currTimesteps < totalTimesteps):
            print("[INFO] CurrTimesteps: %8d" % (currTimesteps))
            currTimesteps += self.runOneEpisode(agent)
            agent.normalNoise.sigma = max(ARG.GaussianNoiseSigmaDecay * agent.normalNoise.sigma, ARG.GaussianNoiseSigma_min)
        self.env.close()
        self.logger.saveHistory()

    def runOneEpisode_evaluate(self, agent: DDPG, maxTimestep, frames: list):
        state = self.env.reset()[0]
        ep_return = 0
        ep_t = 1
        for ep_t in range(1, maxTimestep + 1):
            frames.append(self.env.render())
            stateTensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
            action = agent.actor(stateTensor).item()
            nextState, reward, done, _, _ = self.env.step([action])
            ep_return += reward
            agent.storeTransition(state, action, nextState, reward, done)
            state = nextState
            if (done):
                break
            if (ep_t == maxTimestep):
                break
        print("| Episode reward: %10.4f |" % (ep_return))
        
        return ep_return, ep_t

    def evaluate(self, agent: DDPG, iterNum: int):
        agent.actor.train(False)
        agent.critic.train(False)
        frames = []
        for episode in range(1, iterNum + 1):
            print("Episode: %8d" % (episode))
            ep_return, ep_t = self.runOneEpisode_evaluate(agent, 200, frames)
        saveFramesToGif(frames, "./evaluate.gif")
        self.env.close()

    @property
    def n_action(self):
        return self.env.action_space.shape[0]
    @property
    def n_state(self):
        return self.env.observation_space.shape[0]