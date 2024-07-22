import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from Buffer import Buffer
from PPO import PPO
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
        self.batchBuffer = Buffer()
        self.logger = Logger()
    
    def runBatchEpisode(self, agent: PPO):
        self.batchBuffer.clear()
        ep_totalRewards = []
        t = 0
        while (t < ARG.Max_Timesteps_Per_Batch):
            ep_rewards = []
            state = self.env.reset()[0]
            done = False
            
            for ep_t in range(0, ARG.Max_Timesteps_Per_Episode):
                t += 1
                stateTensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
                action, log_prob = agent.selectAction(stateTensor)
                nextState, reward, done, _, _ = self.env.step(action)
                self.batchBuffer.states.append(state)
                self.batchBuffer.actions.append(action)
                self.batchBuffer.log_probs.append(log_prob)
                self.batchBuffer.dones.append(done)
                ep_rewards.append(reward)
                if (done):
                    break
                if (t >= ARG.Max_Timesteps_Per_Batch):
                    break
                state = nextState
                
            self.batchBuffer.batchLens.append(ep_t + 1)
            self.batchBuffer.rewards.append(ep_rewards)
            ep_totalRewards.append(sum(ep_rewards))
            self.logger.history["reward"].append(sum(ep_rewards))
        return ep_totalRewards
    
    def computeDiscountedRewards(self, batchRewards):
        batchDiscountedRewards = []
        for epRewards in batchRewards[::-1]:
            discountedReward = 0
            for reward in epRewards[::-1]:
                discountedReward = reward + ARG.gamma * discountedReward
                batchDiscountedRewards.append(discountedReward)
        batchDiscountedRewards.reverse()
        return batchDiscountedRewards
    
    def train(self, agent: PPO, totalTimesteps: int):
        currTimesteps = 0
        while (currTimesteps < totalTimesteps):
            ep_totalRewards = self.runBatchEpisode(agent)
            
            currTimesteps += sum(self.batchBuffer.batchLens)
            self.logger["currTimesteps"] = currTimesteps
            self.logger["meanEpisodeLen"] = sum(self.batchBuffer.batchLens) / len(self.batchBuffer.batchLens)
            self.logger["meanEpisodeReward"] = sum(ep_totalRewards) / len(ep_totalRewards)
            
            batchStates = torch.tensor(self.batchBuffer.states, dtype = torch.float32).to(self.device)
            batchActions = torch.tensor(self.batchBuffer.actions).reshape(-1, 1).to(self.device)
            batchLogProbs = torch.tensor(self.batchBuffer.log_probs, dtype = torch.float32).reshape(-1, 1).to(self.device)
            
            batchDiscountedRewards = self.computeDiscountedRewards(self.batchBuffer.rewards)
            batchDiscountedRewards = torch.tensor(batchDiscountedRewards, dtype = torch.float32).to(self.device)
            
            V = agent.critic(batchStates).squeeze()
            # advantages
            A_k = batchDiscountedRewards - V.detach()
            # 標準化優化(幫助收斂，非必要)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-9)
            batchDiscountedRewards = batchDiscountedRewards.reshape(-1, 1)
            A_k = A_k.reshape(-1, 1)
            
            # # 一次訓練某些次數次
            # for _ in range(ARG.NN_Update_Per_Epoch):
            #     V = agent.critic(batchStates)
            #     probs = agent.actor(batchStates)
            #     log_probs = torch.distributions.Categorical(probs).log_prob(batchActions.detach().squeeze())
            #     log_probs = log_probs.reshape(-1, 1)
                
            #     # PPO演算法裡的 pi_theta(a_t|s_t) / pi_theta_k(a_t | s_t)
            #     # 因為log所以除法變減法
            #     ratios = torch.exp(log_probs - batchLogProbs)
            #     # PPO演算法精隨: clamp
            #     loss1 = ratios * A_k
            #     loss2 = torch.clamp(ratios, 1 - ARG.Clip_Ratio, 1 + ARG.Clip_Ratio) * A_k
            #     actorLoss = (-torch.min(loss1, loss2)).mean()
            #     # critic用MSE loss即可
            #     criticLoss = torch.nn.MSELoss()(V, batchDiscountedRewards)
                
            #     agent.actorOptimizer.zero_grad()
            #     actorLoss.backward()
            #     agent.actorOptimizer.step()
                
            #     agent.criticOptimizer.zero_grad()
            #     criticLoss.backward()
            #     agent.criticOptimizer.step()
            
            # mini-batch
            step = batchStates.size(0)
            indexes = np.arange(step)
            minibarchSize = step // ARG.BatchSize
            # 一次訓練某些次數次
            for _ in range(ARG.NN_Update_Per_Epoch):
                np.random.shuffle(indexes)
                for start in range(0, step, minibarchSize):
                    end = start + minibarchSize
                    idx = indexes[start:end]
                    miniStates = batchStates[idx]
                    miniActions = batchActions[idx]
                    miniLogProbs = batchLogProbs[idx]
                    miniAdvantage = A_k[idx]
                    miniDiscountedRewards = batchDiscountedRewards[idx]
                    V = agent.critic(miniStates)
                    probs = agent.actor(miniStates)
                    log_probs = torch.distributions.Categorical(probs).log_prob(miniActions.detach().squeeze())
                    log_probs = log_probs.reshape(-1, 1)
                    # PPO演算法裡的 pi_theta(a_t|s_t) / pi_theta_k(a_t | s_t)
                    # 因為log所以除法變減法
                    ratios = torch.exp(log_probs - miniLogProbs)
                    # PPO演算法精隨: clamp
                    loss1 = ratios * miniAdvantage
                    loss2 = torch.clamp(ratios, 1 - ARG.Clip_Ratio, 1 + ARG.Clip_Ratio) * miniAdvantage
                    actorLoss = (-torch.min(loss1, loss2)).mean()
                    # critic用MSE loss即可
                    criticLoss = torch.nn.MSELoss()(V, miniDiscountedRewards)
                
                    agent.actorOptimizer.zero_grad()
                    actorLoss.backward()
                    agent.actorOptimizer.step()

                    agent.criticOptimizer.zero_grad()
                    criticLoss.backward()
                    agent.criticOptimizer.step()
            
            self.logger.log()
            
        self.logger.saveHistory()
        self.env.close()
        
    def evaluate(self, agent: PPO):
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
    @property
    def actionDim(self):
        return self.env.action_space.n
    @property
    def stateDim(self):
        return self.env.observation_space.shape[0]