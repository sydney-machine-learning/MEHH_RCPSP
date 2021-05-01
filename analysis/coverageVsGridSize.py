import matplotlib.pyplot as plt
x=[125,1000,3375,8000]
y=[66.4,47.1,33.3,23.6]
plt.plot(x,y,'o-')
plt.annotate("125",(125+50,66.4))
plt.annotate("1000",(1000+50,y[1]))
plt.annotate("3375",(3375+50,y[2]))
plt.annotate("8000",(8000+50,y[3]))
plt.xlabel("Grid size")
plt.ylabel("% Coverage")
plt.savefig("../imgs/coverageVsGridSize.png")
plt.show()