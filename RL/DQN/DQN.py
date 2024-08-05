import torch
from QNN import QNN
import ARG
from ReplayBuffer import RelplayBuffer
import numpy as np


class DQN:
    def __init__(self, stateDim, actionDim, device):
        self.Q = QNN(stateDim, actionDim).to(device)
        self.Q_target = QNN(stateDim, actionDim).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.actionDim = actionDim
        self.lr = ARG.lr
        self.gamma = ARG.gamma
        self.epsilon = ARG.initEpsilon
        self.optimizer = torch.optim.Adam(self.Q.parameters(), self.lr)
        self.replayBuffer = RelplayBuffer(stateDim, ARG.BatchSize, ARG.MemoryCapacity)
        self.device = device
        self.lossFunc = torch.nn.MSELoss()
        self.trainCnt = 0
        
    def selectAction(self, state):
        if (np.random.uniform(0, 1) > self.epsilon):
            self.Q.train(False)
            action_Q = self.Q(state)
            action = torch.argmax(action_Q).item()
        else:
            action = np.random.randint(0, self.actionDim)
        return action
    
    def selectAction_evaulate(self, state):
        self.Q.train(False)
        action_Q = self.Q(state)
        action = torch.argmax(action_Q).item()
        return action

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon * ARG.epsilonDecay, ARG.endEpsilon)

    def storeTransition(self, s, a, ns, r, d):
        self.replayBuffer.push(s, a, ns, r, d)

    def train(self):
        self.trainCnt += 1
        self.Q.train(True)
        states, actions, nextStates, rewards, dones = self.replayBuffer.sample()
        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.int64).to(self.device)
        nextStates = torch.tensor(nextStates, dtype = torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype = torch.int32).to(self.device)
        
        Q_eval = self.Q(states).gather(1, actions)
        with torch.no_grad():
            maxQ_next = self.Q_target(nextStates).max(1)[0]
            maxQ_next = maxQ_next.reshape(-1, 1)
            Q_target = rewards + (1 - dones) * self.gamma * maxQ_next
        loss = self.lossFunc(Q_eval, Q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if (self.trainCnt % ARG.Q_Target_Update_Freq == 0):
            self.Q_target.load_state_dict(self.Q.state_dict())
        return loss.item()
    
    def save(self):
        torch.save(self.Q.state_dict(), "./DQN.pth")
    
    def load(self, path: str):
        loaded = torch.load(path)
        self.Q.load_state_dict(loaded)