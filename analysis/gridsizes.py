import matplotlib.pyplot as plt

x=[125,1000,3375,8000]
maxes=[1009.25,1006.34,1007.34,1004.63]
medians=[1003.42,1003.39,1003.22,1003.41]
mins=[1002.36,1001.72,1001.62,1001.47]

plt.plot(x,maxes)
plt.plot(x,medians)
plt.plot(x,mins)
plt.show()