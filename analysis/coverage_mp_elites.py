import pickle
from base_code import *
import matplotlib.pyplot as plt

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
x=list(range(0,26))


plt.plot(x,divs[3],'o-',label='MpEt-M_125')

plt.plot(x,divs[0],'^-',label='MpEt-M_1000')

plt.plot(x,divs[1],'*-',label='MpEt-M_3375')
plt.plot(x,divs[2],'d-',label='MpEt-M_8000')

plt.xlabel('# Generations')
plt.ylabel("Coverage %")
plt.title("Plot of % Coverage vs # generations")

plt.legend()
plt.savefig('../imgs/coverage_plot_mp_elites.png')
plt.show()
