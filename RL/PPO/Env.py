import gym
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib import animation

from Buffer import Buffer
from PPO import PPO

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
            "reward": [],
            "loss": []
        }
        self.device = device
        self.buffer = Buffer()
    
    def runOneEpisode(self, maxScoreLimit: float, agent: PPO):
        state = self.env.reset()[0]
        totalReward = 0
        done = False
        self.buffer.clear()
        while (not done):
            # self.env.render()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = agent.selectAction(state)
            state_next, reward, done, truncated, info = self.env.step(action)
            self.buffer.add(state, )
            totalReward += reward
            agent.ep_rewards.append(reward)
            
            if (done):
                break
            if (totalReward >= maxScoreLimit):
                break
            state = state_next
        discountedAndStandardizedRewards = agent.getDiscountedAndStandardizedRewards()
        loss = agent.train(discountedAndStandardizedRewards)
        self.history["reward"].append(totalReward)
        self.history["loss"].append(loss)
        return loss, totalReward
    
    def runOneEpisode_evaluation(self, maxScoreLimit: float, agent: Agent.Agent, gifPath: str):
        state = self.env.reset()[0]
        totalReward = 0
        done = False
        frames = []
        while (not done):
            frames.append(self.env.render())
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = agent.selectAction_evaluation(state)
            state_next, reward, done, truncated, info = self.env.step(action)
            totalReward += reward
            if (done):
                break
            if (totalReward >= maxScoreLimit):
                break
            state = state_next
        saveFramesToGif(frames, gifPath)
        return totalReward
        
    def train(self, iterNum, maxScoreLimit: float, agent: Agent.Agent):
        for episode in range(1, iterNum + 1, 1):
            loss, reward = self.runOneEpisode(maxScoreLimit, agent)
            print("|Episode: %4d|Loss: %5.6f|Reward: %5.6f" % (episode, loss, reward))
        self.env.close()
        
    def evaluation(self, iterNum, maxScoreLimit: float, agent: Agent.Agent):
        for episode in range(1, iterNum + 1, 1):
            reward = self.runOneEpisode_evaluation(maxScoreLimit, agent)
            print("|Episode: %4d|Reward: %5.6f" % (episode, reward))
        self.env.close()
    
    def log(self, path: str):
        with open(path + "/loss.csv", "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            for loss in self.history["loss"]:
                writer.writerow([loss])
        with open(path + "/reward.csv", "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            for loss in self.history["reward"]:
                writer.writerow([loss])
    
    @property
    def actionDim(self):
        return self.env.action_space.n
    @property
    def stateDim(self):
        return self.env.observation_space.shape[0]