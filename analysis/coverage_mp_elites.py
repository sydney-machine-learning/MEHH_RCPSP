import pickle
from base_code import *
import matplotlib.pyplot as plt
file=open("../logs/gp/set_1/training_logs/training_log_0.txt","r")
lines=file.readlines()
file.close()
div_gp=[]
for i in range(3,len(lines)):
    div_gp.append(100*int(list(lines[i].strip().split())[-1])/1024.0)
data=[]
medians=[1,18,4,19]
divs=[]
for i in range(4):

    file=open("../logs/map_elites/set_{0}/final{1}.p".format(str(i),str(medians[i])),"rb")
    data_temp=pickle.load(file)
    file.close()
    data.append(data_temp)
    logs=list(str(data_temp['logbook']).split('\n'))
    div_tmp=[]
    for i in range(1,len(logs)):
        line=logs[i]
        div_tmp.append((100.0*int(list(list(line.split())[1].split('/'))[0]))/float(list(list(line.split())[1].split('/'))[1]))
    divs.append(div_tmp)
divs.append(div_gp)
x=list(range(0,26))



plt.plot(x,divs[4],'s-',label='GPHH')
plt.plot(x,divs[3],'o-',label='$MEHH_{125}$')
plt.plot(x,divs[0],'^-',label='$MEHH_{1000}$')

plt.plot(x,divs[1],'*-',label='$MEHH_{3375}$')
plt.plot(x,divs[2],'d-',label='$MEHH_{8000}$')

plt.xlabel('# Generations')
plt.ylabel("% Unique individuals")
# plt.title("Plot of % Unique individuals vs # Generations")

plt.legend()
plt.savefig('../imgs/coverage_plot_mp_elites.png')
plt.show()
