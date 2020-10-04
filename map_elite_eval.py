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

from deap import base,creator,tools,algorithms,gp
import statistics
import operator
import instance
import os
import numpy as np
import random
import warnings
import scipy
import instance
import time
from utils import sub_lists
from multiprocessing import Pool
from os import listdir
#Generate the training set

validation_set=[]
for i in range(1,480,10):
    validation_set.append("./RG300/RG300_"+str(i)+".rcp")
print(len(validation_set))

test_set=[]
all_rg300=["./RG300/"+i for i in listdir('./RG300')]
test_set+=[i for i in all_rg300 if i not in validation_set]
print(len(test_set))
# Parameters
from params_map_elites import *


# Update seed



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
def evalSymbReg(individual,train_set):

    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    hard_cases_sum=0
    hard_cases_count=0
    serial_cases_sum=0
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=True)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])
        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
        frac2,makespan2=inst.serial_sgs(option='forward',priority_rule='',priorities=priorities)
        if(inst.rs==0.2 and inst.rf==1.0):
            hard_cases_count+=1
            hard_cases_sum+=frac
        serial_cases_sum+=frac2
    fitness=[sumv/len(train_set)]
    features = [len(individual),hard_cases_sum/hard_cases_count,serial_cases_sum/len(train_set)]
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

best_individual="max(min(mul(max(sub(sub(max(AvgRReq, LF), neg(LS)), mul(div(ES, TPC), sub(MaxRReq, MaxRReq))), mul(add(MinRReq, RR), mul(mul(MinRReq, MinRReq), add(MaxRReq, MaxRReq)))), min(div(LF, div(max(ES, MinRReq), add(LF, LS))), neg(add(max(AvgRReq, MaxRReq), sub(AvgRReq, MaxRReq))))), mul(add(mul(div(sub(MaxRReq, MaxRReq), sub(EF, MinRReq)), min(sub(AvgRReq, TSC), mul(ES, MaxRReq))), div(mul(min(MaxRReq, MinRReq), max(MaxRReq, RR)), max(div(ES, TSC), mul(AvgRReq, LF)))), neg(max(sub(add(EF, LS), mul(ES, MaxRReq)), MinRReq)))), sub(LF, AvgRReq))"

total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(test_set,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
print(total_dev_percent,total_makespan)

# path="./logs/map_elites/set_0/data_and_charts/"

# for run in range(0,31):

#     file=open(path+'grid_'+str(run),"rb")
#     grid=pickle.load(file)
#     min_deviation=100000

#     best_individual=grid.best
#     fin=0    
#     for ind in grid:
        
#         fin+=1
#         total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=ind),mode='parallel',option='forward',use_precomputed=True,verbose=False)
#         print(fin,"/",len(grid),total_dev_percent,total_makespan)
#         if total_dev_percent<min_deviation:
#             min_deviation=total_dev_percent
#             best_individual=ind

#     print(best_individual)
#     file=open(path+"best_func_"+str(run),"wb")
#     pickle.dump(best_individual,file)
#     file.close()

#     total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(test_set,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
#     print("Aggregate % ",total_dev_percent)
#     print("Makespan ",total_makespan )
#     file=open(path+"new_results.txt","a")
#     file.write("Run#"+str(run)+"\n")
#     file.write(str(best_individual)+"\n")
#     file.write("Aggregate% "+str(total_dev_percent)+"\n")
#     file.write("Makespan% "+str(total_makespan)+"\n\n")

#     file.close()