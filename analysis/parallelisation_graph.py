import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

x=[1,2,3,4]
y=[2446,1459,1128,962]
ax = plt.figure().gca()
# plt.title("Plot of time taken vs number of cores (seconds)")
plt.xlabel("Number of cores")
plt.ylabel("Time taken for 5 generations of GP")


ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(x,y,'o-')
plt.savefig("../imgs/parallelisation.png")
plt.show()