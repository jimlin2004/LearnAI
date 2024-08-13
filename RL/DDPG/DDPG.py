import torch as th
import numpy as np
import ARG
from ActorNN import ActorNN
from CriticNN import CriticNN
from ReplayBuffer import RelplayBuffer
from NormalNoise import NormalNoise

class DDPG:
    def __init__(self, n_state, n_action, bound, device):
        self.n_action = n_action
        self.device = device
        self.bound = bound
        
        self.normalNoise = NormalNoise(np.zeros(n_action), ARG.GaussianNoiseSigma * np.ones(n_action))
        
        self.replayBuffer = RelplayBuffer(n_state, ARG.BatchSize, ARG.MemoryCapacity)
        self.actor = ActorNN(n_state, n_action, bound).to(device)
        self.critic = CriticNN(n_state, n_action).to(device)
        self.actor_target = ActorNN(n_state, n_action, bound).to(device)
        self.critic_target = CriticNN(n_state, n_action).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actorOptimizer = th.optim.Adam(self.actor.parameters(), ARG.Actor_lr)
        self.criticOptimizer = th.optim.Adam(self.critic.parameters(), ARG.Critic_lr)

        self.mse = th.nn.MSELoss()
    def selectAction(self, state):
        action = self.actor(state).item()
        action = np.clip(action + self.normalNoise(), -self.bound, self.bound)
        return action
    def storeTransition(self, s, a, ns, r, d):
        self.replayBuffer.push(s, a, ns, r, d)
    # 只更新部分參數
    def softUpdate(self, NN: th.nn.Module, targetNN: th.nn.Module):
        for param, targetParam in zip(NN.parameters(), targetNN.parameters()):
            targetParam.data.copy_(ARG.Tau * param.data  + (1 - ARG.Tau) * targetParam.data)
    def train(self):
        self.actor.train(True)
        self.critic.train(True)
        states, actions, nextStates, rewards, dones = self.replayBuffer.sample()
        states = th.tensor(states, dtype = th.float32).to(self.device)
        actions = th.tensor(actions, dtype = th.float32).to(self.device)
        nextStates = th.tensor(nextStates, dtype = th.float32).to(self.device)
        rewards = th.tensor(rewards, dtype = th.float32).to(self.device)
        dones = th.tensor(dones, dtype = th.int32).to(self.device)
        
        with th.no_grad():
            nextActions = self.actor_target(nextStates)
            q_next = self.critic_target(nextStates, nextActions)
            q_target = rewards + ARG.Gamma * (1 - dones) * q_next
        q = self.critic(states, actions)
        # 加上mean是因為pytorch的MSE沒有取mean
        criticLoss = th.mean(self.mse(q_target, q))
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()
        
        action_eval = self.actor(states)
        scores = self.critic(states, action_eval)
        actorLoss = -th.mean(scores)
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()
        
        # 軟更新targetNN
        self.softUpdate(self.actor, self.actor_target)
        self.softUpdate(self.critic, self.critic_target)
        
        return actorLoss.item(), criticLoss.item()

    def save(self):
        th.save(self.actor.state_dict(), "./saved/Actor.pth")
        th.save(self.critic.state_dict(), "./saved/Critic.pth")
    def load(self, actorPath, criticPath):
        loaded = th.load(actorPath)
        self.actor.load_state_dict(loaded)
        self.actor_target.load_state_dict(self.actor.state_dict())
        loaded = th.load(criticPath)
        self.critic.load_state_dict(loaded)
        self.critic_target.load_state_dict(self.critic.state_dict())