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
import qdpy
import math
validation_set=random.sample(["./"+"RG300"+'/'+i for i in listdir('./'+"RG300") if i!='param.txt'],60)
test_set=[]

train_set= ["./"+"j30"+'/'+i for i in listdir('./'+"j30") if i!='param.txt'] + random.sample(["./"+"j60"+'/'+i for i in listdir('./'+"j60") if i!='param.txt'],80)+random.sample(["./"+"j90"+'/'+i for i in listdir('./'+"j90") if i!='param.txt'],80)
test_set=[]
for typ in ["RG300"]:
    test_set+=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']

POP_SIZE=1024
NUM_GENERATIONS=25
INST_TYPE='j60'
MATING_PROB=0.5
MUTATION_PROB=0.3
SELECTION_POOL_SIZE=7
HOF_SIZE=1
HEIGHT_LIMIT = 6
MU=1024
LAMBDA=1024
GEN_MIN_HEIGHT=3
GEN_MAX_HEIGHT=5
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
    """Evaluation function which calculates fitness"""    
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
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

file=open('./evolved_funcs/gp/evolved_pop_0','rb')
pop=pickle.load(file)
file.close()

min_deviation=100000
# best_individual=grid.best
# print(best_individual)
fin=0    
for ind in pop:
    
    fin+=1
    total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=ind),mode='parallel',option='forward',use_precomputed=True,verbose=False)
    if(fin%50 ==0 ):
        print(fin,"/",len(pop),total_dev_percent,total_makespan)
    if total_dev_percent<min_deviation:
        min_deviation=total_dev_percent
        best_individual=ind
print(best_individual)
total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(test_set,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
print("Aggregate % ",total_dev_percent)
print("Makespan ",total_makespan )