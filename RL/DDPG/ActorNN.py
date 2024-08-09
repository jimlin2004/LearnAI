import torch as th

class ActorNN(th.nn.Module):
    def __init__(self, n_state, n_action, bound):
        super().__init__()
        self.bound = bound
        
        self.l1 = th.nn.Linear(n_state, 32)
        self.relu1 = th.nn.ReLU()
        self.l2 = th.nn.Linear(32, n_action)
        self.tanh1 = th.nn.Tanh()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        # tanh是為了將輸出縮放到[-1, 1]之間，
        # 乘上bound可說放到遊戲的action範圍 -> [-bound, bound]
        out = self.bound * self.tanh1(out)
        return out