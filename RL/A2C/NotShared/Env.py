import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import csv

from Buffer import Buffer
from A2C import A2C
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
        self.buffer = Buffer()
        self.logger = Logger()
        self.history = {
            "timestep": [],
            "reward": []
        }
    
    def runOneEpisode(self, agent: A2C):
        self.buffer.clear()
        ep_t = 0
        state = self.env.reset()[0]
        
        for ep_t in range(1, ARG.Max_Timesteps_Per_Episode + 1):
            stateTensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
            action, log_prob = agent.selectAction(stateTensor)
            nextState, reward, done, _, _ = self.env.step(action)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.dones.append(done)
            self.buffer.rewards.append(reward)
            if (done):
                break
            state = nextState
        return ep_t
    
    def computeDiscountedRewards(self, rewards):
        discountedRewards = np.zeros_like(rewards)
        discountedReward = 0
        for t in reversed(range(len(rewards))):
            discountedReward = rewards[t] + ARG.gamma * discountedReward
            discountedRewards[t] = discountedReward
        return discountedRewards
    
    def train(self, agent: A2C, totalTimesteps: int):
        currTimesteps = 0
        while (currTimesteps < totalTimesteps):
            ep_t = self.runOneEpisode(agent)
            currTimesteps += ep_t
            
            states = torch.tensor(self.buffer.states, dtype = torch.float32).to(self.device)
            actions = torch.tensor(self.buffer.actions, dtype = torch.int32).reshape(-1, 1).to(self.device)
            
            discountedRewards = self.computeDiscountedRewards(self.buffer.rewards)
            discountedRewards = torch.tensor(discountedRewards, dtype = torch.float32).to(self.device)
            
            V = agent.critic(states).squeeze()
            # advantages
            A = discountedRewards - V.detach()
            # 標準化優化(幫助收斂，非必要)
            A = (A - A.mean()) / (A.std() + 1e-9)
            discountedRewards = discountedRewards.reshape(-1, 1)
            A = A.reshape(-1, 1)
            
            V = agent.critic(states)
            probs = agent.actor(states)
            log_probs = torch.distributions.Categorical(probs).log_prob(actions.detach().squeeze())
            log_probs = log_probs.reshape(-1, 1)
            
            actorLoss = (-log_probs * A.detach()).mean()
            # critic用MSE loss即可
            criticLoss = agent.mse(discountedRewards, V)
            
            self.logger["time/current timestep"] = currTimesteps
            self.logger["train/actor loss"] = actorLoss.item()
            self.logger["train/critic Loss"] = criticLoss.item()
            self.logger["rollout/reward"] = sum(self.buffer.rewards)
            self.history["timestep"].append(currTimesteps)
            self.history["reward"].append(sum(self.buffer.rewards))
            self.logger.summary()
            
            agent.actorOptimizer.zero_grad()
            actorLoss.backward()
            agent.actorOptimizer.step()
            
            agent.criticOptimizer.zero_grad()
            criticLoss.backward()
            agent.criticOptimizer.step()
        self.saveHistory()
        self.env.close()
        
    def evaluate(self, agent: A2C):
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
        
    def saveHistory(self):
        with open("saved/history.csv", "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestep", "reward"])
            for t, reward in zip(self.history["timestep"], self.history["reward"]):
                writer.writerow([t, reward])
    
    @property
    def n_action(self):
        return self.env.action_space.n
    @property
    def n_state(self):
        return self.env.observation_space.shape[0]