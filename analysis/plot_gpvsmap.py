import matplotlib.pyplot as plt

mp_10=[17.02346468267907,15.499472236880889,43.17582100629456,1003.3910961781694][:-1]
mp_15=[17.040599588464833, 15.461353505271491, 42.979289562229354,1003.2204131080136][:-1]
mp_20=[16.973159868863423,15.54107349022864, 42.94693521804706,1003.4140062028326][:-1]
mp_5=[16.97613087047303,15.52792482095169,43.15870266355503,1003.418813058263][:-1]
gp=[16.99488829412731,15.54043375576035,43.27991100066281,1004.9214582349127][:-1]
divs=[mp_10,mp_15,mp_20,mp_5]
for i in range(len(divs)):
    divs[i]=[gp[j]-divs[i][j] for j in range(len(gp))]

x=[60,90,120,300][:-1]

plt.plot(x,divs[3],'o-',label='MpEt-M_125')

plt.plot(x,divs[0],'^-',label='MpEt-M_1000')

plt.plot(x,divs[1],'*-',label='MpEt-M_3375')
plt.plot(x,divs[2],'d-',label='MpEt-M_8000')
plt.xticks(x,x)
plt.xlabel('Instance sizes')
plt.ylabel("% improvement over GP")
plt.title("Plot of % improvement over GP vs Size of instance")

plt.legend()
plt.savefig('../imgs/gp_vs_map_elites.png')
plt.show()
