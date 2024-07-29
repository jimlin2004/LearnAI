import matplotlib.pyplot as plt
import numpy as np

XBound = (-32.0, 32.0)
POPULATION_SIZE = 10
GENERATION_NUM = 200
CROSSOVER_RATE = 0.8
MUTATE_RATE = 0.1

def AckleyFunc(x1, x2):
    d = 2
    a, b, c = 20, 0.2, 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt((x1 ** 2 + x2 ** 2) / d))
    term2 = -np.exp((np.cos(c * x1) + np.cos(c * x2)) / d)
    result = term1 + term2 + a + np.e
    return result

def getFitness(population):
    values = AckleyFunc(population[:, 0], population[:, 1])
    # + 1e-5是為了防止除以0的錯誤
    return -(values - np.max(values)) + 1e-5

# # 輪盤法
def select(population, fitness):
    idx = np.random.choice(np.arange(POPULATION_SIZE), size = POPULATION_SIZE * 2, p = fitness / fitness.sum())
    return population[idx]

def crossover(p1, p2):
    alpha = np.random.uniform(0.0, 1.0)
    child = alpha * p1 + (1 - alpha) * p2
    return child

def mutate(individual):
    individual += np.random.uniform(-3.0, 3.0, size = 2)
    np.clip(individual, XBound[0], XBound[1])

if __name__ == "__main__":
    x1 = np.linspace(XBound[0], XBound[1], 1000)
    x2 = x1.copy()
    x1, x2 = np.meshgrid(x1, x2)
    z = AckleyFunc(x1, x2)
    
    history = []
    population = np.random.uniform(XBound[0], XBound[1], size = (POPULATION_SIZE, 2))
    
    for p in range(GENERATION_NUM):
        fitness = getFitness(population)
        parent = select(population, fitness)
        nextPopulation = population.copy()
        for i in range(0, POPULATION_SIZE):
            p1 = parent[2 * i]
            p2 = parent[2 * i + 1]
            child = p1
            if (np.random.uniform(0, 1) <= CROSSOVER_RATE):
                child = crossover(p1, p2)
            if (np.random.uniform(0, 1) <= MUTATE_RATE):
                mutate(child)
            nextPopulation[i] = child
        population = nextPopulation
        history.append(np.max(AckleyFunc(population[:, 0], population[:, 1])))
        print(np.min(AckleyFunc(population[:, 0], population[:, 1])))
        plt.clf()
        plt.contourf(x1, x2, z)
        plt.scatter(population[:, 0], population[:, 1], color = "red")
        plt.show(block = False)
        plt.pause(0.2)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(history)
    plt.show()