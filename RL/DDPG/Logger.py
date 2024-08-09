import csv

class Logger:
    def __init__(self):
        self.history = {
            "reward": [],
            "timestep": [],
            "actor loss": [],
            "critic loss": []
        }
    
    def __getitem__(self, key):
        return self.map[key]
    def __setitem__(self, key, value):
        self.map[key] = value
    
    def saveHistory(self):
        with open("saved/history.csv", "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestep", "reward", "actor loss", "critic loss"])
            for data in zip(self.history["timestep"], self.history["reward"], self.history["actor loss"], self.history["critic loss"]):
                writer.writerow(list(data))