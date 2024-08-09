Gamma = 0.99
GaussianNoiseSigma = 0.01 # gaussian noise 的標準差
Tau = 0.005 # soft update 係數

Actor_lr = 3e-4
Critic_lr = 3e-4

Max_Timesteps_Per_Episode = 200

BatchSize = 256

# 在正式開始前需要先經過多少timestep
initReplayBufferSize = 1000
MemoryCapacity = 1000000