import pickle
from base_code import *
import matplotlib.pyplot as plt
file=open("../logs/gp/set_1/training_logs/training_log_0.txt","r")
lines=file.readlines()
file.close()
div_gp=[]
for i in range(3,len(lines)):
    div_gp.append(int(list(lines[i].strip().split())[-1]))
file=open("../logs/map_elites/set_0/final0.p","rb")
data=pickle.load(file)
file.close()
logs=list(str(data['logbook']).split('\n'))
div_mp=[]
for i in range(1,len(logs)):
    line=logs[i]
    div_mp.append(int(list(list(line.split())[1].split('/'))[0]))
print(div_gp)
print(div_mp)
x=list(range(0,26))
plt.plot(x,div_gp,'o-',label='GP-M')
plt.plot(x,div_mp,'^-',label='MpEt-M_1000')
plt.xlabel('# Generations')
plt.ylabel("Unique population size")
plt.title("Plot of unique population size vs # generations")

plt.legend()
plt.savefig('../imgs/diversity_plot.png')
plt.show()
