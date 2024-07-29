import numpy as np

class RelplayBuffer:
    def __init__(self, stateDim, batchSize, bufferSize):
        self.batchSize = batchSize
        self.maxSize = bufferSize
        # ReplayBuffer的pointer，代表現在要寫入的位置
        self.ptr = 0
        self.currSize = 0
        
        self.state = np.zeros(shape = (self.maxSize, stateDim), dtype = np.float32)
        self.action = np.zeros(shape = (self.maxSize, 1), dtype = np.int64)
        self.nextState = np.array(self.state, dtype = np.float32)
        self.reward = np.zeros(shape = (self.maxSize, 1), dtype = np.float32)
        self.done = np.zeros(shape = (self.maxSize, 1), dtype = np.int32)
        
    def push(self, state, action, nextState, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr][0] = action
        self.nextState[self.ptr] = nextState
        self.reward[self.ptr][0] = reward
        self.done[self.ptr][0] = done
        
        self.ptr = (self.ptr + 1) % self.maxSize
        self.currSize = min(self.currSize + 1, self.maxSize)
        
    def sample(self):
        index = np.random.randint(0, self.currSize, size=self.batchSize)
        return (
            self.state[index],
            self.action[index],
            self.nextState[index],
            self.reward[index],
            self.done[index]
        )
    def __len__(self):
        return self.currSize