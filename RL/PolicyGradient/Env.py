import gym
import Agent
import torch
import numpy as np
import csv

class Env:
    def __init__(self, device, gameName: str, renderMode = "human"):
        self.env = gym.make(gameName, render_mode = renderMode)
        self.history = {
            "reward": [],
            "loss": []
        }
        self.device = device
    
    def runOneEpisode(self, maxScoreLimit: float, agent: Agent.Agent):
        state = self.env.reset()[0]
        totalReward = 0
        done = False
        agent.ep_rewards.clear()
        agent.log_probs.clear()
        while (not done):
            self.env.render()
            # state = np.array(state[0])
            state = torch.from_numpy(state).to(self.device)
            action = agent.selectAction(state)
            state_next, reward, done, truncated, info = self.env.step(action)
            totalReward += reward
            agent.ep_rewards.append(reward)
            
            if (done):
                break
            if (totalReward >= maxScoreLimit):
                break
            state = state_next
        discountedAndNormalizedRewards = agent.getDiscountedAndNormalizedRewards()
        loss = agent.train(discountedAndNormalizedRewards)
        self.history["reward"].append(totalReward)
        self.history["loss"].append(loss)
        return loss, totalReward
        
    def train(self, iterNum, maxScoreLimit: float, agent: Agent.Agent):
        for episode in range(1, iterNum + 1, 1):
            loss, reward = self.runOneEpisode(maxScoreLimit, agent)
            print("|Episode: %4d|Loss: %5.6f|Reward: %5.6f" % (episode, loss, reward))
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