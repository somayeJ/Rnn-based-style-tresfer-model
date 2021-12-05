in the bi_gru model: we did not reverse the seq here and had x+ padding as opposed to reversed (padding + x) 

in attention model, we have rev seq and have padding at first(it is the same as mono_gru_model)


this code is based on the folder attention_bidirectional_gru, and we do these changes based on Bahadanau'paper and this link(https://www.tensorflow.org/tutorials/text/nmt_with_attention):

	1. in nn.tf.v.1.py  file, line 283, I correct and fed the output of thee GRU to the attention layer instead of h

	2. use z as the first output state for t=0, to compute the context_vector (https://www.tensorflow.org/tutorials/text/nmt_with_attention)

	3. in this code https://www.tensorflow.org/tutorials/text/nmt_with_attention (which is based on Bahadanau'paper) the decoder is initialized randomly, I do the same

	4. softmax in the code https://www.tensorflow.org/tutorials/text/nmt_with_attention and Marc's code are the same, since both softmaxes are applied to the number of the input

	5. ***** in all codes and also this one, we use #src_contexts =  tf.concat([src_hs_fw[:,:,100:], src_hs_bw[:,:,100:]],axis=2), we can change it in the next run, as a hyperparameter change
	5.5.***** we did do this hyperparameter tunnng && used #src_contexts = tf.concat([src_hs_fw[:,:,:], src_hs_bw[:,:,:]],axis=2), we can change it in the next run, as a hyperparameterchange

	6. ***** In this code, we consider the first output state for t=0, as z and make z like the following (we use self.z2):
		self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1) 
		self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)
	we can change it in the next run, as a hyperparameter to self.z0 = tf.concat([src_h_fw, src_h_bw] , axis=1)

	7. if we want to change hyperparameters in 5&6, we should be consistent and change both
	
	8. Check the indices in attention



It seems like Marc's method is different from Bahadanau'paper and this link  (https://www.tensorflow.org/tutorials/text/), so in the next version, we run by changing towards Bahadanau' version
???. check the masking method again
???. check the two_layer feed forward layer, implemented in my code and https://www.tensorflow.org/tutorials/text/nmt_with_attention


***********immediate next step: change the hyperparameter in no 6

---------adv_SDS_attention.py && adv_SDS_attention_backup_v1.py: These two codes are almost the same as each other, 

---------adv_SDS_attention_backup_v2.py: I put the loop of generation out of the graph in the session section, but it did not work

***************************************************

The folder in the tmp file, attention_biGru_v2.1, is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py, where for the encoder outputs at each step, the first 100 dimention is disregarded &&  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)  is considered as the first output token at first step of generation

The results in attention_biGru_v2.2 in the tmp folder is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py in  the folder, attention_biGru_v2.1 , where based on the full zat each step of the enocer context_veector is built  &&  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1) which is the firt token at first step of generation


The folder in the tmp file,  attention_biGru_v2.3  is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py && has these differences with attention_biGru_v2.2:
		1.   we consider the first output state for t=0, as z, 	self.z0 = tf.concat([src_h_fw, src_h_bw]
		where as in attention_biGru_v2.2, z as the first token is:self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)
		2. z in beam_search file also changes and we call model.z0 as the self.z0 = tf.concat([src_h_fw, src_h_bw]

		



	
