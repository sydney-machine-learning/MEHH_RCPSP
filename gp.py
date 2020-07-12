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
from utils import read_param,add_lists,sub_lists, less_than, min_finish_time, find_index #Utility functions
import numpy as np
from deap import base,creator,tools,algorithms,gp
import operator
import math
train_set=[]

types=['j30','j60']
for typ in types:
  for i in range(1,49):
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")

POP_SIZE=1024
NUM_GENERATIONS=25
INST_TYPE='j60'
MATING_PROB=0.5
MUTATION_PROB=0.3
SELECTION_POOL_SIZE=7
HOF_SIZE=3
HEIGHT_LIMIT = 2
MU=1000
LAMBDA=200
def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN",10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
counts=0
def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    
    func = toolbox.compile(expr=individual)
    sumv=0
    for i in range(len(train_set)):
        
        file=train_set[i]
        
        inst=instance.instance(file,use_precomputed=True)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])

        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
    
    return (sumv/len(train_set),)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=3, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)
pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(HOF_SIZE)
pop, log = algorithms.eaMuPlusLambda(pop, toolbox,MU,LAMBDA, MATING_PROB, MUTATION_PROB, NUM_GENERATIONS, stats=mstats,halloffame=hof, verbose=True)


for hof_index in range(HOF_SIZE):
    nodes, edges, labels = gp.graph(hof[hof_index])
    print("Function ", hof[hof_index])
    test_type=['j30','j60','j90','j120']
    for typ in test_type:
        total_dev_percent,makespan=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=hof[hof_index]),inst_type=typ,mode='serial',option='forward')
        print(typ,total_dev_percent,makespan)
        log_file=open('results_log.txt','a+')
        log_file.write(str(hof[hof_index])+" : \n              "+INST_TYPE+"         "+typ+"         "+str(len(train_set))+"               "+str(NUM_GENERATIONS)+"          "+str(MATING_PROB)+"           "+str(MUTATION_PROB)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n\n")
        log_file.close()



        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
    
        g.draw("./gp_trees/"+INST_TYPE+"__"+str(hof[hof_index])+"__"+str(round(total_dev_percent,2))+".png")