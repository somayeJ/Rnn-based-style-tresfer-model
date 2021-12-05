
import numpy as np
import scipy.stats
import sys
from ci import mean_confidence_interval

def process_list(a):
    
    #print(a[:3], type(a[0]))
    a1=[float(i.split(',')[1].strip()) for i in a]
    a2=[round(i,4) for i in a1]
    #print(a2[:3])
    return a2



def read_file(files):
    with open(files[0], 'r') as f0, open(files[1], 'r') as f1:
        a =f0.readlines()
        aa=process_list(a)
        b=f1.readlines()
        bb= process_list(b)
        c= aa+bb
    return c


file_names = sys.argv[1:]
#print(file_names)
data = read_file(file_names)

h,standard_error,mian,mian_minus,mian_plus = mean_confidence_interval(data,confidence=0.99)
print('h: which is the amount deducted and added to mian,standard_error,mian,mian_minus_h,mian_plus_h',h,standard_error,mian,mian_minus,mian_plus)
#print(mean_confidence_interval(data))