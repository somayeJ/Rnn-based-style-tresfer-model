#the code used is for sample mean.  this calculation is for sample mean, so a t distribution is used. If the questions is to calculation population mean, a normal distribution should be used and the confident interval will be smaller for the same confidence level. 

import numpy as np
import scipy.stats


# my_mean_confidence_interval: idid some modifications
# taken from this site https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=1):
    a = 1.0 * np.array(data)
    n = len(a)
    # se: standard erorr
    # The standard error of a statistic is the standard deviation of its sampling distribution or an estimate of that standard deviation. If the 
    # statistic is the sample mean, it is called the standard error of the mean.
    m, se = np.mean(a), scipy.stats.sem(a)
    # ppf: Percent point function (inverse of cdf (Cumulative distribution function.)— percentiles).
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h,se,m, m-h, m+h


# The "Population Standard Deviation uses n in the formula of standard deviation
# The "Sample Standard Deviation uses n-1 in the formula of standard deviation
# with n refering to the no of samples



#h,se,m,mm,mp=my_mean_confidence_interval(a)
#print(h,se,m,mm,mp)