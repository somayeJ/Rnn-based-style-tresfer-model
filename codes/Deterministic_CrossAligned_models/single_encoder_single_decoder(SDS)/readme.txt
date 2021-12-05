in attention model, we have rev seq and have padding at first(it is the same as mono_gru_model)
*************************************************************************
0. bi_gru model : we did not reverse the seq here and had x+ padding as opposed to reversed (padding + x) 


*************************************************************************************************************

1. attention_biGru v1 :
	* In this files the output where Rnn method, src_context is concatenated with outputs at each step 
	* in tmp file, attention_biGru v1/test1 is the output of running the codes in the folder, this code with these hyperparam:
			1. in nn.tf.v.1.py file, line 283,  h is fed to attention layer instead of output
			2.  the generator is initialized with z 
			3. in self.h_ori, line 121, was not used reuse=True in the linear method


*****************************************************************************************************************
2.attention_biGru_v2, based on Bahadanau'paper and this link(https://www.tensorflow.org/tutorials/text/nmt_with_attention) in the sense that the context vector is concatenated to the input token, but the feed forward layers are based on Marc's paper: 
	
	* the following three hyperparameter tuning are considered

					2.1     ********** The results in "attention_biGru_v2.1", is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py,  in  the folder, attention_biGru_v2, where at each step of z of encoder the first 100 dimention is disregarded and context_vector is built accordingly &&  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)


					2.2	********** The results in "attention_biGru_v2.2" is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py in the folder, attention_biGru_v2 , where based on the full z at each step of the encoder context_vector is built
						 and z as the 0th generated of the decoder is:         self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)


					2.3	********** The folder in the tmp file,  "attention_biGru_v2.3"  is as a result of running, adv_SDS_attention.py && nn_tf_v_1.py && has these differences with attention_biGru_v2.2:
								1.   we consider the first output state for t=0, as z ,	self.z0 = tf.concat([src_h_fw, src_h_bw]
								where as in attention_biGru_v2.2, z as the first token is:         self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)

					

*************************************************************************************************************************
3. In the attention_biGru_v3 folder, 
Version a: it is fully  based on the  Bahadanau'paper and this link(https://www.tensorflow.org/tutorials/text/nmt_with_attention)
	context_vector is built based on the full z at each step of the enocer as the (values: encoder_step outputs) and  z as the 0th output state for t=0, is computed as self.z0 = tf.concat([src_h_fw, src_h_bw] (code is saved in attention_biGru_v3 folder && outputs in tmp it is saved in attention_biGru_v3 folder/hyp3)

the following hyperparameters can be tuned:

		1.  where at each step of z the first of encoder the first 100 dimention is disregarded &&  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)  is considered as the first output token at first step of generation&& self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1) 

		2. where  context_vector is built based on the full z at each step of the enocer as the (values: encoder_step outputs) and  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)  as the firt output token at first step of generation&& self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1) 

(
Version a: it is fully  based on the  Bahadanau'paper and this link(https://github.com/jhyuklee/nmt-pytorch):
		the difference is in intialization of the weights and also removing the bias of the last layer
		
