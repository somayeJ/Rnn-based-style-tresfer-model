in the bi_gru model : we did not reverse the seq here and had x+ padding as opposed to reversed (padding +x) 

in attention model, it is the same as mono_gru_model, rev seq and have padding at first

in tmp, test2 compared to test1, in line 175:
	1. in nn.tf.v.1.py  file, line 283,I correct and add the output to the attention layer instead of h
	2. I  initialize the generator with random vector instead of z , line119 , there , i did the next step too
	3. in self.h_ori, line 121, concat init_zeros and label in the linear method, then consequently the same for h_tsf


