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
from numpy import exp

os.environ["CUDA_VISIBLE_DEVICES"]="1"

file_write_ent_att_w_each_seq =  '../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.test.ent_att_w_each_seq'

att_vect0_masked=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testattention_w.0.npy',allow_pickle=False )
att_vect1_masked=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testattention_w.1.npy',allow_pickle=False )

len_src0=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testsrc_len.0.npy',allow_pickle=False )
len_src1=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testsrc_len.1.npy',allow_pickle=False )

len_g0=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testgen_len.0.npy',allow_pickle=False )
len_g1=np.load('../probing_classification_experiments/data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_v1_3.2.2/sentiment.testgen_len.1.npy',allow_pickle=False )
'''
att_vect0_masked=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testattention_w.0.npy',allow_pickle=False )
att_vect1_masked=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testattention_w.1.npy',allow_pickle=False )

len_src0=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testsrc_len.0.npy',allow_pickle=False )
len_src1=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testsrc_len.1.npy',allow_pickle=False )

len_g0=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testgen_len.0.npy',allow_pickle=False )
len_g1=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/yelp/rerun_v1_3.2.2/sentiment.testgen_len.1.npy',allow_pickle=False )


att_vect0_masked=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testattention_w.0.npy',allow_pickle=False )
att_vect1_masked=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testattention_w.1.npy',allow_pickle=False )

len_src0=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testsrc_len.0.npy',allow_pickle=False )
len_src1=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testsrc_len.1.npy',allow_pickle=False )

len_g0=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testgen_len.0.npy',allow_pickle=False )
len_g1=np.load('./data/upsampling_emb/attention_based/attention_biGru_v3/gyafc_preprocessed/rerun_3.2.2/sentiment.testgen_len.1.npy',allow_pickle=False )

'''

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

def pad_func(src_len_seq,max_src_len=30):
	data_size = len(src_len_seq)
	src_pad_weights= []
	for len_sent in (src_len_seq):
		print(len_sent)
		w_each_seq=[]
		for i in range(int(max_src_len-len_sent)):
			w_each_seq.append(0.0)
		for i in range(int(len_sent)):
			w_each_seq.append(1.0)
		src_pad_weights.append([w_each_seq]*max_src_len)
	src_mask = np.reshape(src_pad_weights,[data_size,max_src_len,max_src_len])
	return src_mask

def softmax(vector):
	e = exp(vector)
	return e / e.sum()
def max_ent(len_seq):
	'''
	'''
	prob_seq = ([1/len_seq]*int(len_seq))
	return scipy.stats.entropy(prob_seq, base=2)

def seq_att_w_entopy(src_len_seq,src_masked_att,gen_len_seq, file_write_ent_att_w_each_seq,mode,max_src_len=30):


	#1:max ent for input seqs
	all_att_entropy=[]

	#2  Calculate attention entropy for any output token of the output sequence  , && divide each token attention entropy by the corresponding max_ent
	for att_w_seqs,len_gen,len_src  in zip(src_masked_att,gen_len_seq,src_len_seq):
		print('len_gen,len_src,,att_w_seqs',len_gen,len_src,att_w_seqs)
		#att_w_seqs.shape =max_output_len* max_src_len, but max_output_ is masked based on the len_gen, so if len_gen for this att_w_seqs is 2 it means that 
		#the first 2 rows are non-zero and the rest are 0s
		softmax_att_w_each_output_seq =[]
		for l in range(int(len_gen)):
			att_ws_each_output_token = [att_w_seqs[l][int(max_src_len-(len_src+i))] for i in reversed(range(int(len_src)))]

			softmax_att_w_each_output_token = softmax(att_ws_each_output_token)
			softmax_att_w_each_output_seq.append(softmax_att_w_each_output_token)
			#print(['att_w_seqs[l][int(max_src_len-len_src)]],',att_w_seqs[l][int(max_src_len-len_src)]])
		print('softmax_w_output_each_seq',softmax_att_w_each_output_seq)

		normalized_ent_att_w_output_each_seq = [(scipy.stats.entropy(softmax_w_vec, base=2))/float(max_ent(len_src)) for softmax_w_vec  in softmax_att_w_each_output_seq if len_src>1]
		n= np.average(normalized_ent_att_w_output_each_seq)
		print('normalized_ent_att_w_output_each_seq',normalized_ent_att_w_output_each_seq,'n',n,'len-src',len_src,len_gen)
		if len_src>1:
			all_att_entropy.append(n)
		print('all_att_entropy',all_att_entropy)
		print('normalized_ent_att_w_output_each_seq',normalized_ent_att_w_output_each_seq)
		print('all_att_entropy',all_att_entropy)
		print('float(max_ent(len_src))',float(max_ent(len_src)))
	np.save(file_write_ent_att_w_each_seq+mode+'.npy',all_att_entropy)
	return  np.average(all_att_entropy)

def normalizing_att_z_seqs(att_w_seqs,src_len_seq,gen_len_seq,mode):
	'''
	att_w_seqs.shape=[data.shape,20,20], it is a masked vector based on the len_gen, for instance if the len_gen is 2, from vect3 to 20 are 0 vectors

	'''
	normalalized_seq = []
	src_masked_w =  pad_func(src_len_seq)
	src_masked_att = att_w_seqs*src_masked_w 
	'''
	print('att_w_seqs_masked0',att_w_seqs[0])
	print('src_masked_w[0]',src_masked_w[0])
	print('src_masked_att',src_masked_att[1])

	print('att_w_seqs_masked1',att_w_seqs[2])
	print('src_masked_w[1]',src_masked_w[2])
	print('src_masked_att2',src_masked_att[2])
	print('gen_len_seq',gen_len_seq[:3])
	print('src_len_seq',src_len_seq[:3])
	'''
	#the average of the  seq_attention_entropy assigned to each seq
	avg_ent_each_seq_all_data =seq_att_w_entopy(src_len_seq,src_masked_att,gen_len_seq,file_write_ent_att_w_each_seq,mode)
	return avg_ent_each_seq_all_data

'''
yelp
#w_seq[:len_g][len_src:]
avg_ent_each_seq_all_data = normalizing_att_z_seqs(att_vect1_masked,len_src1,len_g1)
print(avg_ent_each_seq_all_data)#0.999577425478   
avg_ent_each_seq_all_data = normalizing_att_z_seqs(att_vect0_masked,len_src0,len_g0)
print(avg_ent_each_seq_all_data)# 0.999350135581  


GYAFC preprocessed
avg_ent_each_seq_all_data = normalizing_att_z_seqs(att_vect0_masked,len_src0,len_g0)
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.99971312410938051 , 0.99767788045650352  

avg_ent_each_seq_all_data = normalizing_att_z_seqs(att_vect1_masked,len_src1,len_g1)
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.9977565311829415   
'''
mode = '.1'
avg_ent_each_seq_all_data = normalizing_att_z_seqs(att_vect1_masked,len_src1,len_g1,mode)# mode and the suffixes of the files should be the same
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.9977565311829415   
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.99971312410938051 , 0.99767788045650352      
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.99971312410938051 , 0.99767788045650352      
print('avg_ent_each_seq_all_data',avg_ent_each_seq_all_data)#0.99971312410938051 , 0.99767788045650352      +