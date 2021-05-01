import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir
import pickle
import time
from sympy import *
from sympy.parsing.sympy_parser import *
from sympy import Add,Mul,div,Min,Max

import statistics
import numpy as np
from deap import base,creator,tools,algorithms,gp
import operator
import math
import time 
import multiprocessing
import re

files = ["gp_data","map_elites_data_3","map_elites_data_0","map_elites_data_1","map_elites_data_2"]

def div(left, right): # Safe division to avoid ZeroDivisionError
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Generate the primitive set which contains all operators
pset = gp.PrimitiveSet("MAIN",10)
pset.addPrimitive(operator.add, 2,'+')
pset.addPrimitive(operator.sub, 2,'-')
pset.addPrimitive(operator.mul, 2,'*')
pset.addPrimitive(div, 2,'/')
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
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit size of operator tree
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches
def graph(best_individual,filename):
    ind = creator.Individual.from_string(best_individual,pset)
    # Generate and Store graph
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("../gp_trees/best/"+filename+".png")
def clean(x):
   
    
    x = x.replace('neg','-')
    x = x.replace('add','Add')
    x = x.replace('mul','Mul')
    x = x.replace('mul','Mul')
    x = x.replace('min','Min')
    x = x.replace('max','Max')
    x = x.replace("RR","RRq")
    # print(x)
    lis = list(find_all(x,"sub"))
    for i in lis:
        j = i+4
        ctr = 0
        while j <len(x):
            if(x[j] =='('):
                ctr+=1
            elif x[j] ==')':
                ctr-=1
            elif x[j]==',' and ctr ==0:
                x = x[:j+1]+"-"+x[j+1:]
                break
            j+=1
    
    lis = list(find_all(x,"div"))
    # print(lis)
    for i in lis:
        j = i+4
        ctr = 0
        second = False
        while j <len(x):
            if(x[j] =='('):
                ctr+=1
            elif x[j] ==')':
                
                if(ctr == 0 and second):
                    x = x[:j]+')'+x[j:]
                    break
                ctr-=1
            elif x[j]==',' and ctr ==0:
                x = x[:j+1]+"1/("+x[j+1:]
                j+=3
                second=True
            j+=1
    x = x.replace('sub','Add')
    x = x.replace('div','Mul')
    return x
for ind in range(len(files)):
    file = open("../"+files[ind],"rb")
    data = pickle.load(file)
    file.close()
    mindev=1000000
    index = 0
    bestrule = data[0]['ind']
    for i in range(31):
        if(data[i]['dev'] <mindev):
            bestrule = data[i]['ind']
            mindev = data[i]['dev']
            index = i
    
    bestrule = clean(bestrule)
    # bestrule = clean("div(AvgRReq,y)")
    # print(bestrule)
    print(mindev)
    
    x =  str(parsing.sympy_parser.parse_expr(bestrule,evaluate=True))
    x =x.replace("RRq","RR")
    print(x)
    # graph(bestrule,files[ind])


"""
-(-(*(LS, LF), +(TSC, neg(neg(AvgRReq)))), *(*(*(max(*(LF, ES), min(LS, RR)), *(min(max(MinRReq, EF), *(ES, MinRReq)), min(max(LS, EF), min(TSC, AvgRReq)))), *(*(-(/(MaxRReq, TSC), *(LS, MaxRReq)), min(max(MinRReq, RR), -(MinRReq, TSC))), *(-(*(MaxRReq, EF), *(TPC, MaxRReq)), -(/(RR, MaxRReq), /(LS, MaxRReq))))), -(+(+(min(max(TPC, TSC), -(RR, MinRReq)), *(neg(MaxRReq), min(AvgRReq, EF))), neg(neg(min(RR, ES)))), min(/(min(1, /(AvgRReq, TSC)), +(*(TPC, EF), +(MinRReq, MinRReq))), neg(-(min(TPC, AvgRReq), AvgRReq))))))
"""