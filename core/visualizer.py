import matplotlib.pyplot as plt

def plot_ga_log(log):
    gen = log.select("gen")
    max_f = log.select("max")
    avg_f = log.select("avg")

    plt.figure()
    plt.plot(gen, max_f, label="Max fitness")
    plt.plot(gen, avg_f, label="Avg fitness")
    plt.xlabel("Generation")
    plt.ylabel("Performance per Watt")
    plt.title("GA Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
