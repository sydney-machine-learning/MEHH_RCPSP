import matplotlib.pyplot as plt
import numpy as np
mp_10=[17.02346468267907,15.499472236880889,43.17582100629456,1003.3910961781694][:]
mp_15=[17.040599588464833, 15.461353505271491, 42.979289562229354,1003.2204131080136][:]
mp_20=[16.973159868863423,15.54107349022864, 42.94693521804706,1003.4140062028326][:]
mp_5=[16.97613087047303,15.52792482095169,43.15870266355503,1003.418813058263][:]
gp=[16.99488829412731,15.54043375576035,43.27991100066281,1004.9214582349127][:]
divs=[mp_10,mp_15,mp_20,mp_5]
for i in range(len(divs)):
    divs[i]=[gp[j]-divs[i][j] for j in range(len(gp))]

# x=[60,90,120,300][:]

# plt.plot(x,divs[3],'o-',label='$MEHH_{125}$', markersize=5)

# plt.plot(x,divs[0],'^-',label='$MEHH_{1000}$', markersize=5)

# plt.plot(x,divs[1],'*-',label='$MEHH_{3375}$', markersize=5)
# plt.plot(x,divs[2],'d-',label='$MEHH_{8000}$', markersize=5)
# plt.xticks(x,x)
# plt.xlabel('Instance sizes')
# plt.ylabel("% improvement over GPHH")
# # plt.title("Plot of % improvement over GPHH vs Size of instance")

# plt.legend()
# plt.savefig('../imgs/gp_vs_map_elites.png')
# plt.show()


labels = ['60', '90', '120', '300']



x = np.arange(len(labels))*1.2  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects4 = ax.bar(x - 3 * width/2-0.06, divs[3], width, label='$MEHH_{125-M}$')
rects1 = ax.bar(x - width/2-0.04, divs[0], width, label='$MEHH_{1000-M}$')
rects2 = ax.bar(x + width/2-0.02, divs[1], width, label='$MEHH_{3375-M}$')
rects3 = ax.bar(x + 3*width/2, divs[2], width, label='$MEHH_{8000-M}$')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Improvement in deviation %')
ax.set_xlabel("Instance size")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if(height<0 and height>=-0.01):
            height=0.00
        print(height)
        offset = -10 if height<0 else 3
        y = height if height < 0 else height
        ax.annotate("{:.2f}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, y),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize = 8)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.axhline(0,color='k',linestyle="dashed",linewidth=1)
plt.savefig('../imgs/gp_vs_map_elites.png')
plt.show()