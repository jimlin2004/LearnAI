import torch
import Env
import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = Env.Env(device, "CartPole-v1", "rgb_array")
    agent = Agent.Agent(device, env.stateDim, env.actionDim)
    agent.loadModel("./loadModel/policyGradientModel.pth")
    env.runOneEpisode_evaluation(1000, agent, "evaluation.gif")
    plt.plot(list(range(1, len(env.history["reward"]) + 1)), env.history["reward"])
    plt.show()