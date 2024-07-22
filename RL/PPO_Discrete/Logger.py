import csv

class Logger:
    def __init__(self):
        self.map = {
            "currTimesteps": 0,
            "meanEpisodeLen": 0,
            "meanEpisodeReward": 0
        }
        self.history = {
            "reward": []
        }
    
    def __getitem__(self, key):
        return self.map[key]
    def __setitem__(self, key, value):
        self.map[key] = value
    
    def log(self):
        print("---------------------------------------------")
        for k, v in self.map.items():
            print("| %20s | %18.6f |" % (k, v))
        print("---------------------------------------------")
    
    def saveHistory(self):
        with open("reward.csv", "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            for reward in self.history["reward"]:
                writer.writerow([reward])