from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import csv

class Callback(BaseCallback):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.episodeRewards = []
        self.currEpisodeReward = 0
    def _on_step(self) -> bool:
        if (self.locals["dones"].item()):
            self.currEpisodeReward += self.locals["rewards"][0]
            self.episodeRewards.append(self.currEpisodeReward)
            self.currEpisodeReward = 0
        else:
            self.currEpisodeReward += self.locals["rewards"][0]
        return True

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs = 1)
vec_env.envs[0].env._max_episode_steps = 1024

# newLogger = configure("baselineLog/", ["stdout", "csv"])

model = PPO("MlpPolicy", vec_env, verbose = 1)
# model.set_logger(newLogger)
callback = Callback(1)
model.learn(total_timesteps = 102400, callback=callback)
episodeRewards = callback.episodeRewards

with open("baseline_reward.csv", "w", newline = "") as csvfile:
    writer = csv.writer(csvfile)
    for reward in episodeRewards:
        writer.writerow([reward])

plt.plot(episodeRewards)
plt.show()