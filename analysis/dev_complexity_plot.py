import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir

import numpy as np
from deap import base,creator,tools,gp
import operator

import time 
import pickle
from adjustText import adjust_text

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

files = ["gp_data","map_elites_data_3","map_elites_data_0","map_elites_data_1","map_elites_data_2"]

rules = ['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD','IRSM','ACS','WCS','$GPHH_B$','$MEHH_{125-B}$','$MEHH_{1000-B}$','$MEHH_{3375-B}$','$MEHH_{8000-B}$']
devs=[1020.19, 1028.43, 1020.43, 1014.39, 1077.04, 1016.3, 1013.72, 1071.38, 1044.91, 1081.61, 1079.58, 1023.52, 1050.69]
complexities=[1,1,1,1,1,1,1,0,4,4,19,24,18]
disp=[(5.035282,1024.939571),
(0.604234,1034.241536),
(8.235484,1021.123381),
(6.758468,1014.206536),
(6.019960,1077.412190),
(15.866734,1018.738262),
(3.312097,1008.482250),
(4.050605,1066.917667),
(11.189516,1044.497548),
(10.943347,1083.136476),
(28.913710,1081.228381),
(32.360081,1024.462548),
(25.713508,1052.129929),
(5.281452,1001.803917),
(25.221169,1008.482250),
(94.887097,1007.051179),
(60.423387,1007.289690),
(44.422379,1016.353143)]

removals=["GRD","SPT","RAND","IRSM"]
removals=[]
for i in removals:
    index=rules.index(i)
    rules.pop(index)
    devs.pop(index)
    complexities.pop(index)
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
    
    print(mindev)
    # bestrule=toolbox.compile(expr=bestrule)
    count=0
    unique_prefixes=["RR","LS","LF","ES","EF","TSC","TPC","add","sub","max","min","mul","div","neg"]
    for x in unique_prefixes:
        count+=bestrule.count(x)
    print(bestrule)
    complexities.append(count)
    devs.append(float(mindev))
    print(files[ind],count)


fig, ax = plt.subplots()
ax.scatter(complexities, devs)

texts=[]
for i, txt in enumerate(rules):
    # ax.annotate(rules[i],(complexities[i],devs[i]))
    # texts.append(plt.text(complexities[i],devs[i],rules[i],fontsize=12,verticalalignment='top'))
    ax.annotate(rules[i], (complexities[i], devs[i]), xytext=(disp[i][0],disp[i][1]), arrowprops = dict(arrowstyle="->",color="red"))

# adjust_text(texts, complexities,devs, arrowprops=dict(arrowstyle="->", color='r', lw=1.0),force_points=(1,1),va="top",expand_text=(1.1,1.5),expand_points=(1.1,1.5),expand_objects=(1.2,1.2))
plt.xlabel("Complexity (No. of Nodes)")
plt.ylabel("Performance on test set (% deviation)")
plt.savefig("../imgs/complexity_plot.png")
def onclick(event):
    print('(%f,%f)' %
          ('double' if event.dblclick else event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()