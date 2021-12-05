
This code is based on adv_SDS_attention.py && nn_tf_v_1.py of the folder attention_biGru_gru_v2, but the  difference is the following is that:
	********the two_layer feed forward layer, implemented in this code is based on the code and https://www.tensorflow.org/tutorials/text/nmt_with_attention where as in adv_SDS_attention.py && nn_tf_v_1.py of the folder attention_bidirectional_gru_v3,  it is based on  Marc's method 

*****************************************************************
the following hyperparameters can be tuned
	1.  where at each step of z the first of encoder the first 100 dimentio is disregarded &&  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)  is considered as the firt output token at first step of generation&& self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1) 

	2. where  context_vector is built based on the full z at each step of the enocer as the (values: encoder_step outputs) and  self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)  as the firt output token at first step of generation&& self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1) 

	3. where  context_vector is built based on the full z at each step of the enocer as the (values: encoder_step outputs) and  z as the first output state for t=0, is computed as self.z0 = tf.concat([src_h_fw, src_h_bw]
		


	
		



	
