import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import pandas as pd
import numpy as np
plt.style.use('default')
from matplotlib import rcParams
config = {
    "font.family": 'serif', # 衬线字体
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
s=52
e=62
ctg = np.load("/home/ljr/keyan/Time-Series-Library-main/time_adp.npy")[s:e,s:e]
plt.figure(dpi=1200)
ax = sns.heatmap(ctg, annot=False, cmap='RdPu', fmt=".2f",linewidths=0.5)  
ax.xaxis.tick_top()  # Move x-axis labels to top

ax.set_xticklabels([str(i) for i in range(s,e)], rotation=0,
                     fontproperties='Times New Roman')
ax.set_yticklabels([str(i) for i in range(s,e)], rotation=0,
                     fontproperties='Times New Roman')
plt.title("C-C-T Graph",fontproperties='Times New Roman')
plt.xlabel("C",fontproperties='Times New Roman')
plt.ylabel("C",fontproperties='Times New Roman')
plt.show()
plt.savefig("traffic.png")