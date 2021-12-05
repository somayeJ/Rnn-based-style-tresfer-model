import numpy as np
import random
from sklearn import preprocessing
x_DS= [ 0.721, 0.743,0.85, 0.726, 0.833, 0.238,0.231, 0.225,0.197, 0.725,  0.71,  0.743]
standardized_X_DS = preprocessing.scale(x_DS)
#print(standardized_X_DS)
scaled_data_DS = preprocessing.minmax_scale(x_DS)
#print(scaled_data_DS)
def minmaxscale(vector):
	print(len(vector),vector,'***********')
	print(preprocessing.minmax_scale(vector),'***********')
	print('std', np.std(vector),'***********')
	return

#minmaxscale(x_DS)
x_DS_merge =[ 0.471,0.466,0.67,0.498,0.68,0.3,0.295,0.225, 0.263,0.453,0.461,0.559]
#minmaxscale(x_DS_merge)

sv =[ 0.241,0.263,0.32,0.263,0.32, 0.141,0.136, 0.132,0.119,0.272,0.282,0.314]
#minmaxscale(sv)

sv_merge = [ 0.136,0.152,0.262,0.149,0.275,0.079,0.078,0.083, 0.06,0.16,0.148,0.3]
#minmaxscale(sv_merge)

aligned = [ 0.77,0.751,0.793,0.729,0.8,0.516, 0.476, 0.515,0.451,0.739,0.731,0.847]

elmo =[ 0.74,0.76,0.87,0.75,0.86,0.23,0.3,0.22,0.29,0.74,0.74]
md=[ 0.607, 0.665,0.797,0.625,0.756,0.187,0.131,0.172,0.1,0.63,0.592,0.594]
md_merge=[  0.573, 0.651,0.757,0.6,0.673, 0.293,0.371,0.312, 0.398,0.574,0.531,0.53]
mv =[ 0.238,0.254,0.306,0.249,0.322,0.106,0.113, 0.107,0.076,0.243,0.251,0.22]
mv_merge =[ 0.252, 0.28,0.325,0.3,0.332,0.066,0.094,0.078,0.099,0.278,0.262, 0.287]
minmaxscale(mv_merge)