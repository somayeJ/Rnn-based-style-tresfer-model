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

if __name__=="__main__":
    path_content_preserve_scores_file =  sys.argv[1] # ../model_outputs/AE_ : path to the file of semantic resemblance between phrases and generated files
    path_generated_file = sys.argv[2] # ../model_outputs/AE/sentiment.test. :path to the generated file
    path_orig_file = sys.argv[3]  # ../../../data/yelp/sentiment.test.  :path to the original file
    file_suffix = sys.argv[4] #'rec' ot 'txt' :generated file suffix
    i=0
    for index_name in ["0", "1"]:
        content_preservation_file = path_content_preserve_scores_file +  "style" + index_name + "_semantics.txt"
        file_orig = path_orig_file + index_name
        generated_file = path_generated_file +index_name +file_suffix
        with open(content_preservation_file, 'r') as c, open(file_orig,'r') as o, open(generated_file) as g:
            content_scores = c.readlines()
            orig_lines = o.readlines()
            gen_lines =g.readlines()
            if i == 0 :
                len_orig0 =[len(line.split()) for line in orig_lines]
                content_scores_float0 = [round(float(score),2) for score in content_scores ]
                len_gen0 = [len(line.split()) for line in gen_lines]
            else:
                len_orig1 =[len(line.split()) for line in orig_lines]
                content_scores_float1 = [round(float(score),2) for score in content_scores ]
                len_gen1 = [len(line.split()) for line in gen_lines]
        i+=1
    data0 = len_gen0 + len_gen1
    data1 = len_orig0 + len_orig1
    data2 = content_scores_float0 + content_scores_float1

    print(data2[:10])



    corr02, _ = pearsonr(data0, data2)
    print('Pearsons correlation gen length and scores: %.3f' % corr02)

    corr12, _ = pearsonr(data1, data2)
    print('Pearsons correlation orig length and scores: %.3f' % corr12)
    pyplot.scatter(data1, data2,marker='o')

    pyplot.show()

