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
import scipy.stats
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"
'''

att_vect0_masked=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testattention_w.0.npy',allow_pickle=False )
att_vect1_masked=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testattention_w.1.npy',allow_pickle=False )

len_src0=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testsrc_len.0.npy',allow_pickle=False )
len_src1=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testsrc_len.1.npy',allow_pickle=False )

len_g0=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testgen_len.0.npy',allow_pickle=False )
len_g1=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testgen_len.1.npy',allow_pickle=False )
'''

att_vect0_masked=np.load('./sentiment.testattention_w.0.npy',allow_pickle=False )
att_vect1_masked=np.load('./sentiment.testattention_w.1.npy',allow_pickle=False )

len_src0=np.load('./sentiment.testsrc_len.0.npy',allow_pickle=False )
len_src1=np.load('./sentiment.testsrc_len.1.npy',allow_pickle=False )

len_g0=np.load('./sentiment.testgen_len.0.npy',allow_pickle=False )
len_g1=np.load('./sentiment.testgen_len.1.npy',allow_pickle=False )

#print('length of written files 0 and their seq embedding size',len(vectors0), len(vectors0[0]))
#print('length of written files 1 and their seq embedding size',len(vectors1), len(vectors1[0]))
print('len(att_vect000)',att_vect0_masked.shape,att_vect0_masked[0][0])
print('len(att_vect001)',att_vect0_masked.shape,att_vect0_masked[0][1])
print('len(att_vect1)',att_vect1_masked.shape,att_vect1_masked[0][-1])
print('(len_src0)0',len_src0[0])
print('(len_g0)0',len_g0[0])

print('***************************************************************************')
print('len(len_src0)',len_src0.shape)
print('len(len_src1)',len_src1.shape)

print('len(len_g0)',len_g0.shape)
print('len(len_g1)',len_g1.shape)
'''
data = [0.25,0.25,0.25,0.25]
data1 = [0.0,0.15,0.8,0.05]
data2 = [0.0,0,0,0]
for i,j,k in zip(data,data1,data2):
	print(i,j,k)

pd_series = pd.Series(data)
counts = pd_series.value_counts()
entropy00_data = scipy.stats.entropy(data,base=2)
entropy01_pd_series = scipy.stats.entropy(pd_series,base=2)

print(counts)
print('entropy00_data',entropy00_data)
print('entropy01_pd_series',entropy01_pd_series)
pd_series1 = pd.Series(data1)
counts1 = pd_series1.value_counts()
entropy10 = scipy.stats.entropy(data1,base=2)
entropy11 = scipy.stats.entropy(pd_series1,base=2)
print(counts1)
print('entropy10',entropy10)
print('entropy11',entropy11)
'''
prob_inputs, entropy_inputs= [],[]
for r in len_src0:
	prob_inputs.append([1/r]*int(r))
for prob_seq in prob_inputs:
	entropy_inputs.append(scipy.stats.entropy(prob_seq, base=2))
print(len(entropy_inputs))

def normalizing_att_z_seqs(att_w_seqs,src_len_seq,gen_len_seq):
	'''
	att_w_seqs.shape=[data.shape,20,20], it is a masked vector based on the len_gen, for instance if the len_gen is 2, from vect3 to 20 are 0 vectors

	'''
	normalalized_seq = []
	for w_seq,len_src,len_g in zip(att_w_seqs,src_len_seq,gen_len_seq):
		print('att_w_seqs_masked0',att_w_seqs[0])
		print('att_w_seqs_masked1',att_w_seqs[1])
		print('att_w_seqs_masked2',att_w_seqs[2])
		print('gen_len_seq',gen_len_seq[:3])
		print('src_len_seq',src_len_seq[:3])
		input()
		normal_gen_w_seq=w_seq[:int(len_g)]
		print(w_seq)
		print(len_g,w_seq[0])
		print(len_g,w_seq[1])
		print(w_seq.shape,normal_gen_w_seq.shape)

		print(len_g)
		print('src',len_src,int(len_src))
		print(len_src.shape,src_len_seq.shape)
		#print('w_seq',w_seq)
		print(w_seq[:int(len_g)][int(len_src):])
		print(w_seq.shape)
		print(w_seq[0])
		print([0])
		print(w_seq[-1])
		#w_seq[:len_g][len_src:]
normalizing_att_z_seqs(att_vect0_masked,len_src0,len_g0)
print(len_g0[0],att_vect0_masked[0])

print(len_g0[1],att_vect0_masked[1])

print(len_g0[10],att_vect0_masked[10])