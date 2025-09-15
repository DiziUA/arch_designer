import random
from deap import base, creator, tools, algorithms
from .architecture import Architecture

class Optimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def random_search(self, n_iter=100):
        best = None
        best_score = -float('inf')
        for _ in range(n_iter):
            arch = Architecture.random_architecture()
            m = self.simulator.simulate(arch)
            score = m.get('perf_per_watt_ml', m['perf_per_watt'])
            if score > best_score:
                best, best_score = (arch, m), score
        return best

    def genetic_algorithm(self, pop_size=50, ngen=40, cxpb=0.5, mutpb=0.2):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        arch_types = ['CPU','GPU','DSP','FPGA','ASIC']
        cache_types = ['Inclusive','Exclusive','Non-inclusive']

        toolbox.register("arch_type_attr", random.choice, arch_types)
        toolbox.register("pipeline_attr", random.choice, [4,8,12,16,20])
        toolbox.register("cache_type_attr", random.choice, cache_types)
        toolbox.register("cache_size_attr", random.choice, [1,2,4,8,16])
        toolbox.register("compute_units_attr", random.randint, 1,64)
        toolbox.register("branch_pred_attr", random.choice, ['Static','Dynamic'])
        toolbox.register("ooo_attr", random.choice, [0,1])

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.arch_type_attr, toolbox.pipeline_attr,
                          toolbox.cache_type_attr, toolbox.cache_size_attr,
                          toolbox.compute_units_attr, toolbox.branch_pred_attr,
                          toolbox.ooo_attr), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_arch(ind):
            arch = Architecture(
                arch_type=ind[0],
                pipeline_stages=ind[1],
                cache_type=ind[2],
                cache_size=ind[3],
                compute_units=ind[4],
                branch_predictor=ind[5],
                out_of_order=bool(ind[6])
            )
            m = self.simulator.simulate(arch)
            return (m.get('perf_per_watt_ml', m['perf_per_watt']),)

        toolbox.register("evaluate", eval_arch)
        toolbox.register("mate", tools.cxTwoPoint)

        def mutate_ind(ind):
            idx = random.randrange(len(ind))
            if idx == 0:
                ind[0] = random.choice(arch_types)
            elif idx == 1:
                ind[1] = random.choice([4,8,12,16,20])
            elif idx == 2:
                ind[2] = random.choice(cache_types)
            elif idx == 3:
                ind[3] = random.choice([1,2,4,8,16])
            elif idx == 4:
                ind[4] = random.randint(1,64)
            elif idx == 5:
                ind[5] = random.choice(['Static','Dynamic'])
            else:
                ind[6] = random.choice([0,1])
            return ind,

        toolbox.register("mutate", mutate_ind)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(f[0] for f in x)/len(x))
        stats.register("min", lambda x: min(f[0] for f in x))
        stats.register("max", lambda x: max(f[0] for f in x))

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen,
                                       stats=stats, halloffame=hof, verbose=True)

        best_ind = hof[0]
        best_arch = Architecture(
            arch_type=best_ind[0],
            pipeline_stages=best_ind[1],
            cache_type=best_ind[2],
            cache_size=best_ind[3],
            compute_units=best_ind[4],
            branch_predictor=best_ind[5],
            out_of_order=bool(best_ind[6])
        )
        best_metrics = self.simulator.simulate(best_arch)
        return best_arch, best_metrics, log
