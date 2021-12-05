# this code is to create the embeddings as the last state of the encoder, and takes train, dev and test data and with two styles saves them into files with their numpy format
# it can be run in the terminal by running a code like the following:
# python ...z_create.py   --load_model true --model ../../tmp/.../model  --vocab ../../tmp/.../yelp.vocab  --output ../probing_classification/output_emb/adv_MD/sentiment.train  --train ../../data/yelp/sentiment.train                                    
# --train or --dev or -- test shows the directory of the files we want to produce their embedding  representation, --output shows where want to store the files with embedding representations
# --model and --vocab: shows the address of the directory in which  model and the vocab file is stored from which we want to compute the emb_rep of the sequences 
# for different models, we need to replace the model part of that model in this file which is  from line 183 up and keep the rest
import os
import sys
import time
import ipdb
import random
import cPickle as pickle
#import _pickle  as pickle
import numpy as np
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"]="0"



vectors0=np.load('sentiment.testcnt_vect.0'+'.npy',allow_pickle=False )
vectors1=np.load('sentiment.testcnt_vect.1'+'.npy',allow_pickle=False )

vec0=np.load('sentiment.testattention_w.0'+'.npy',allow_pickle=False )
vec1=np.load('sentiment.testattention_w.1'+'.npy',allow_pickle=False )

len0=np.load('sentiment.testsrc_len.0'+'.npy',allow_pickle=False)
len1=np.load('sentiment.testsrc_len.1'+'.npy',allow_pickle=False)

len_gen0=np.load('sentiment.testgen_len.0'+'.npy',allow_pickle=False)
len_gen1=np.load('sentiment.testgen_len.1'+'.npy',allow_pickle=False)


#print('length of written files 0 and their seq embedding size',len(vectors0), len(vectors0[0]))
#print('length of written files 1 and their seq embedding size',len(vectors1), len(vectors1[0]))
print(vec0)
print('sentiment.test_attention_w.0masked',len(vec0))
print('sentiment.test_attention_w.0masked',len(vec0), len(vec0[0]), vec0[0][17])
print('sentiment.testcnt_vect.0',len(vectors0), len(vectors0[0]),vectors0[0][0])
print('sentiment.testlen.0',len0)
print('sentiment.testcntlen_gen.0',len_gen0)




