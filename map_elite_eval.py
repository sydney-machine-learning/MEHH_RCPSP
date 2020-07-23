
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pygraphviz as pgv
from qdpy.algorithms.deap import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
import statistics
import operator
import instance
import os
import numpy as np
import random
import warnings
import scipy
import instance
import pickle



train_set=[]

types=['j30','j60']
for typ in types:
  for i in range(1,49):
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")

# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
# Parameters
nb_features = 4                            # The number of features to take into account in the container
nb_bins = [20,7,20,20]
features_domain = [(1, 63),(1,7),(0.7,1.2),(0.1,0.4)]      # The domain (min/max values) of the features
fitness_domain = [(0., 1.0)]               # The domain (min/max values) of the fitness
init_batch_size = 1024                     # The number of evaluations of the initial batch ('batch' = population)
batch_size = 1024                           # The number of evaluations in each subsequent batch
nb_iterations = 15                        # The number of iterations (i.e. times where a new batch is evaluated)
cxpb = 0.8
mutation_pb = 0.2                        # The probability of mutating each value of a genome
max_items_per_bin = 3                      # The number of items in each bin of the grid
verbose = True                             
show_warnings = True                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
log_base_path = '.'

SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HEIGHT_LIMIT = 7 # Height Limit for tree
GEN_MIN_HEIGHT=1
GEN_MAX_HEIGHT=5
"""Eval mode
0:  Evaluate only best individual on train set
1: Evaluate best individual on validation set
2: Evaluate all individuals on grid
"""

eval_mode=[0,1]
def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def evalSymbReg(individual):
    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    hard_cases_sum=0
    hard_cases_count=0
    serial_cases_sum=0
    for i in range(len(train_set)):
        inst=instances[i]
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j],inst.grpw[j],inst.grd[j])
        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
        frac2,makespan2=inst.serial_sgs(option='forward',priority_rule='',priorities=priorities)
        if(inst.rs==0.2 and inst.rf==1.0):
            hard_cases_count+=1
            hard_cases_sum+=frac
        serial_cases_sum+=frac2
    fitness=[sumv/len(train_set)]

    
    hard_cases_frac=hard_cases_sum/hard_cases_count
    serial_cases_frac=serial_cases_sum/len(train_set)
    length = len(individual)
    height = individual.height
    features = [length,individual.height,hard_cases_frac,serial_cases_frac]
    # print(features)
    return [fitness, features]
    # return[fitness,[len(individual),individual.height]]

pset = gp.PrimitiveSet("MAIN",12)
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
pset.renameArguments(ARG10="GRPW")
pset.renameArguments(ARG11="GRD")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

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

with open("final.p", "rb") as f:
    data = pickle.load(f)


print(data.keys())
print(data['container'].best)

grid=data['container']
test_type=['j30','j60','j90','j120']
sum_total_dev=0
sum_counts=0
print("Individual : ", grid.best)
log_file=open('results_log.txt','a+')
for typ in test_type:
    total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=grid.best),inst_type=typ,mode='parallel',option='forward',verbose=False)
    print(typ,total_dev_percent,makespan)
    
    log_file.write(str(grid.best)+" : \n               "+typ+"         "+str(len(train_set))+"               "+str(nb_iterations)+"          "+str(cxpb)+"           "+str(mutation_pb)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n")
    
    sum_total_dev+=total_dev
    sum_counts+=count
log_file.write("Aggregate % : "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
log_file.close()
print("Aggregate % ",(sum_total_dev*100)/sum_counts)
