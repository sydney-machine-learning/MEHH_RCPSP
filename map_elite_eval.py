
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
with open("final.p", "rb") as f:
    data = pickle.load(f)
print(data)
