import pickle
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pygraphviz as pgv
from qdpy.algorithms.deap import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *

from deap import base,creator,tools,algorithms,gp
import statistics
import operator
import os
import numpy as np
import random
import warnings
import scipy

import time
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from multiprocessing import Pool
from os import listdir
matplotlib.use('TKAgg',warn=False, force=True)
path="../logs/map_elites/set_1/"
train_set=['../datasets/j30/'+i for i in listdir('../datasets/j30') if i!="param.txt"]
validation_set=[]
for i in range(1,480,10):
    validation_set.append("../datasets/RG300/datasets/RG300_"+str(i)+".rcp")
all_rg300=["../datasets/RG300/"+i for i in listdir('../datasets/RG300')]
test_set=[i for i in all_rg300 if i not in validation_set]
def div(left, right): # Safe division to avoid ZeroDivisionError
    try:
        return left / right
    except ZeroDivisionError:
        return 1
n_runs=31
nb_features = 3                            # The number of features to take into account in the container
nb_bins = [15,15,15]
features_domain = [(4, 127),(0,30),(1.65,2.00)]      # The domain (min/max values) of the features
fitness_domain = [(0., 1.0)]               # The domain (min/max values) of the fitness
init_batch_size = 1024                     # The number of evaluations of the initial batch ('batch' = population)
batch_size = 1024                           # The number of evaluations in each subsequent batch
nb_iterations = 25                       # The number of iterations (i.e. times where a new batch is evaluated)
cxpb = 0.8
mutation_pb = 0.2                        # The probability of mutating each value of a genome
max_items_per_bin = 1                      # The number of items in each bin of the grid
verbose = True                             
show_warnings = True                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds

SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HEIGHT_LIMIT = 7 # Height Limit for tree
GEN_MIN_HEIGHT=2
GEN_MAX_HEIGHT=5
"""Eval mode
0:  Evaluate only best individual on train set
1: Evaluate best individual on validation set
2: Evaluate all individuals on grid
"""
eval_mode=[0,1]

occupied=0 # Set number for storing results
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
def evalSymbReg(individual,train_set):

    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    hard_cases_sum=0
    hard_cases_count=0
    serial_cases_sum=0
    total_slack=0
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=True)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])
        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
        total_slack+=inst.slack/inst.n_jobs
    
    # str_ind=str(individual)
    # prec_count=str_ind.count("ES")+str_ind.count("EF")+str_ind.count("LS")+str_ind.count("LF")+str_ind.count("TPC")+str_ind.count("TSC")
    total_slack/=len(train_set)
    fitness=[sumv/len(train_set)]
    features = [len(individual),str(individual).count("RR"),total_slack]
    # print(individual)
    return [fitness, features]



# Toolbox defines all gp functions such as mate,mutate,evaluate
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg,train_set=train_set)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit size of operator tree
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))


i=18
sz=15
file=open(path+"final"+str(i)+".p","rb")
data=pickle.load(file)

file.close()
# print(data['container'].fitness[(9,6,2)][0][0])
grid=np.zeros((sz,sz*sz))
mask=np.zeros((sz,sz*sz))
print(data['container'].fitness)
for j in data['container'].fitness:
    print(j)    
    
    if(data['container'].fitness[j]):
        print(j[0],j[1]+sz*j[2],(data['container'].fitness)[j][0].values)
        grid[j[0]][j[1]+sz*j[2]]=float(data['container'].fitness[j][0].values[0])
    else:
        mask[j[0]][j[1]+sz*j[2]]=1
        grid[j[0]][j[1]+sz*j[2]]=0
yticks=list(range(0,30,3))


with sns.axes_style("white"):
    # sns.set(rc={ 'axes.facecolor':'black'})
    cmap = sns.cm.rocket
    f, ax = plt.subplots(figsize=(20, 2))
    ax = sns.heatmap(grid, mask=mask,square=True,cmap=cmap)
    ax.invert_yaxis()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    xt=range(0,sz*sz,sz)
    xlis=range(0,sz*sz,sz)
    yt=range(sz)
    ylis=range(4,127,13)
    plt.xlabel("Slack",fontsize=12)
    plt.ylabel("Number of nodes")
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.xaxis.set_minor_formatter(plt.FuncFormatter(format_func2))
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
    ax.tick_params(axis ='both', which ='both', length = 10)
    plt.tick_params(axis='x', which='minor', labelsize=12)
    plt.tick_params(axis='x', which='minor', labelsize=10)
    # plt.yticks(yt,ylis)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title("Performance heat map",fontsize=16)
    plt.savefig("../imgs/performancegrid_15.png",bbox_inches='tight')
    plt.show()


