in attention model, we have rev seq and have padding at first(it is the same as mono_gru_model)
*************************************************************************
0. bi_gru model: we did not reverse the seq here and had x+ padding as opposed to reversed (padding + x) 


*************************************************************************************************************

1. cyc_loss_plus_base_model_v0: 
computing the recycle loss in seq2seq model, by feeding the encoder with the token_seq_soft_tsf is the seq of the tf.matmul(softmax(logit), embedding) where logits are the outputs of the gru cell at each step of generation after feeding them to a projection layer

2. cyc_loss_plus_base_model_v1: 
computing the recycle loss in seq2seq model, by feeding the encoder with the seq_soft_tsf as the seq of the softmax(logit) where logits are the outputs of the gru cell at each step of generation after feeding them to a projection layer

 
3. cyc_loss_plus_base_model_v2: 
changing the computation of the recycle loss in seq2seq model, towards the method of the transformer



********** cyc_loss_plus_base_model_v1 && cyc_loss_plus_base_model_v2 do not work since, we we need to do similar to cyc_loss_plus_base_model_v0 which is similar to what is done in transformer model
