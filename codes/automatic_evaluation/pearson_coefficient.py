#encoding=utf-8
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sys
import numpy as np
from scipy import spatial
import random
from scipy.stats import pearsonr
from matplotlib import pyplot


#y_data0 = [ 0,0.025,0.0197,-0.0072,-0.0093,-0.0295,-0.0353,-0.0374]
#y_data1 = [ 0,-15.51,-15.65,2.5,2.73,24.77,27.4,28.47]
'''
y_data0 = [-0.0093,-0.0072,-0.0374,-0.0353, -0.0295,0,0.0197,0.025]
y_data1 = [2.73, 2.5,28.47,27.4,24.77,0,-15.65,-15.51]

g_data0 = [ 0,0.0013,0.0068,0.0109,0.0116,0.0163,0.0246,0.0695]
g_data1 = [ 0,-1.54,-2.32,-5.5,-18.35,-8.78,-10.42,-32.73]

y_data0 = [-0.0093,-0.0072,-0.0374,-0.0353, -0.0295,0,0.0197]
y_data1 = [2.73, 2.5,28.47,27.4,24.77,0,-15.65]

g_data0 = [ 0,0.0013,0.0068,0.0109,0.0116,0.0246,0.0695]
g_data1 = [ 0,-1.54,-2.32,-5.5,-18.35,-10.42,-32.73]

y_data0 = [0.8989,0.9239,0.9311,0.9542]
y_data1 = [67.25,99.97,100,100]

g_data0 = [0.8922,0.8848,0.9072,0.9085]
g_data1 = [ 59.69,100,99.93,99.42]

y_data0 = [0.8989,0.9239,0.9311,0.9542]
y_data1 = [67.25,99.97,100,100]

g_data0 = [0.8848,0.8922,0.9072,0.9085]
g_data1 = [ 100,59.69,99.93,99.42]

y_data0 = [0.8989,0.9042,0.9239,0.9311,0.9542]
y_data1 = [67.25,98.39,99.97,100,100]

g_data0 = [0.8848,0.8922,0.8969,0.9072,0.9085]
g_data1 = [ 100,59.69,99.17,99.93,99.42]
'''
# SDM, SDS, MDS, att_SDS


y_data0 = [-0.0093,-0.0072,-0.0374,-0.0353, -0.0295,0,0.025]
y_data1 = [2.73, 2.5,28.47,27.4,24.77,0,-15.51]

g_data0 = [ 0,0.0013,0.0068,0.0109,0.0163,0.0246,0.0695]
g_data1 = [ 0,-1.54,-2.32,-5.5,-8.78,-10.42,-32.73]


corr02, _ = pearsonr(y_data0, y_data1)
print('Pearsons correlation between style and content yelp: %.3f' % corr02)

corr12, _ = pearsonr(g_data0, g_data1)
print('Pearsons correlation between style and content GYAFC: %.3f' % corr12)
pyplot.scatter(y_data0, y_data1,marker='o')

pyplot.show()

