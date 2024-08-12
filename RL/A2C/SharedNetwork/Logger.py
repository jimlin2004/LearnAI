class Logger(object):
    def __init__(self):
        self.data = dict()
    def __getitem__(self, key: str):
        splittedKeys = key.split("/")
        i = 0
        key_len = len(splittedKeys)
        currDict = self.data
        while (i < key_len - 1):
            if (splittedKeys[i] not in currDict):
                raise KeyError()
            currDict = currDict[splittedKeys[i]]
            i += 1
        return currDict[splittedKeys[i]]
    def __setitem__(self, key: str, value):
        splittedKeys = key.split("/")
        i = 0
        key_len = len(splittedKeys)
        currDict = self.data
        while (i < key_len - 1):
            if (splittedKeys[i] not in currDict):
                currDict[splittedKeys[i]] = dict()
                currDict = currDict[splittedKeys[i]]
            else:
                currDict = currDict[splittedKeys[i]]
            i += 1
        currDict[splittedKeys[i]] = value
    def printData_recursive(self, currDict: dict, padding: int):
        for k, v in currDict.items():
            if (isinstance(v, dict)):
                print("| %*s%-*s | %12s |" % (padding, "", (20 - padding), k + '/', ""))
                self.printData_recursive(v, padding + 2)
            else:
                print("| %*s%-*s | " % (padding, "", (20 - padding), k), end = "")
                if (isinstance(v, int)):
                    print("%-12d |" % (v))
                elif (isinstance(v, float)):
                    print("%-12f |" % (v))
                elif (isinstance(v, str)):
                    print("%-12s |" % (v))
    def summary(self):
        print(" -------------------------------------")
        self.printData_recursive(self.data, 0)
        print(" -------------------------------------")