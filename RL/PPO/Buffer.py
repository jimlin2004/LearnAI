class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.nextStates = []
        self.rewards = []
        self.dones = []
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.nextStates.clear()
        self.rewards.clear()
        self.dones.clear()
    def add(self, s, a, ns, r, d):
        self.states.append(s)
        self.actions.append(a)
        self.nextStates.append(ns)
        self.rewards.append(r)
        self.dones.append(d)