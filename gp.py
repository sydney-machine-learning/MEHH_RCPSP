import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir
import pickle
import time
import instance
import statistics
import numpy as np
from deap import base,creator,tools,algorithms,gp
import operator
import math
import time
#Generate the training set
train_set=[]
types=['j30','j60']
for typ in types:
  for i in range(1,49):
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")

# Parameters for GP
POP_SIZE=1024
NUM_GENERATIONS=25 # Number of generation to evolve
MATING_PROB=0.5 # Probability of mating two individuals
MUTATION_PROB=0.3 # Probability of introducing mutation
SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HOF_SIZE=1 # Number of top individuals to evaluate on test set
HEIGHT_LIMIT = 6 # Height Limit for tree
MU=1024 # The number of individuals to select for the next generation.
LAMBDA=1024 # The number of children to produce at each generation.
GEN_MIN_HEIGHT=3
GEN_MAX_HEIGHT=5


def div(left, right): # Safe division to avoid ZeroDivisionError
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Generate the primitive set which contains all operators
pset = gp.PrimitiveSet("MAIN",10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
#Rename all arguments
pset.renameArguments(ARG0="ES")
pset.renameArguments(ARG1="EF")
pset.renameArguments(ARG2="LS")
pset.renameArguments(ARG3="LF")
pset.renameArguments(ARG4="TPC")
pset.renameArguments(ARG5="TSC")
pset.renameArguments(ARG6="RR")
pset.renameArguments(ARG7="AvgRReq")
pset.renameArguments(ARG8="MaxRReq")
pset.renameArguments(ARG9="MinRReq")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Define the Fitness type(minimisation) 
# weights=(-1,) indicates that there is 1 fitness value which has to be minimised
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Define Individual type (PrimitiveTree)

def evalSymbReg(individual):
    """
        Evaluation function which calculates fitness of an individual
        Parameters: 
            individual : The individual whose fitness is being evaluated
        Returns:
            A tuple of fitness values since only 1 value is used we return a 1 element tuple (fitness,)
    """
    start=time.time()
    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=True)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])

        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
    return (sumv/len(train_set),)

# Toolbox defines all gp functions such as mate,mutate,evaluate
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit size of operator tree
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))

# Statistics calculated by evaluating GP
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

# Update seed
seed = np.random.randint(1000000)
np.random.seed(seed)
random.seed(seed)

# Initialise population and run algo
pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(HOF_SIZE)
"""
    1. eaSimple - At each gen both crossover and mutation are applied on population with probability 0.9 and 0.1,
                and the population is replaced by the offspring
    >>> 2. eaMuPlusLambda - At each gen either crossover or mutation is applied with probability 0.9 and 0.1,
                        and x number of children are produced and the new population is formed by, 
                        choosing y children from n+x(previous population+offspring)
    3. eaMuCommaLambda - Same as above except the new population is formed from only the x offspring by choosing y of them 
"""
pop, log = algorithms.eaMuPlusLambda(pop, toolbox,MU,LAMBDA, MATING_PROB, MUTATION_PROB, NUM_GENERATIONS, stats=mstats,halloffame=hof, verbose=True)
print(count)
#Store the hof in a pickled file
file=open('./evolved_funcs/best_funcstest','wb')
hof=pickle.dump(hof,file)
file.close()


# Evaluate on test set and generate tree structure
for hof_index in range(HOF_SIZE):


    
    print("Function ", hof[hof_index])
    test_type=['j30','j60','j90','j120']
    sum_total_dev=0
    sum_counts=0
    for typ in test_type:
        total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=hof[hof_index]),inst_type=typ,mode='parallel',option='forward')
        print(typ,total_dev_percent,makespan)
        log_file=open('results_log.txt','a+')
        log_file.write(str(hof[hof_index])+" : \n               "+typ+"         "+str(len(train_set))+"               "+str(NUM_GENERATIONS)+"          "+str(MATING_PROB)+"           "+str(MUTATION_PROB)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n\n")
        log_file.close()
        sum_total_dev+=total_dev
        sum_counts+=count
    print("Aggregate % ",(sum_total_dev*100)/sum_counts)



    # Generate and Store graph
    nodes, edges, labels = gp.graph(hof[hof_index])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("./gp_trees/"+str(round(total_dev_percent,2))+"__3.png")
