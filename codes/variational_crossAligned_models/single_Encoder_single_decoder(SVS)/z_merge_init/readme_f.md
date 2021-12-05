style_transfer_VAE_merge_z.py is based on style_transfer_VAE_1.py and zs (output of encoder) are merged with decoder outputs before feeding to softmax layer

In the paper,and also here we use the method of z_merge where just z is merged (not label+z)

codes which are changed for in this approach: style_transfer_VAE_merge_z.py, beam_search_tf_1.py and nn_tf_1.py and utils_tf_1.py





training loss 
dev loss 31.30, rec 28.06, adv 3.24, d0 0.66, d1 0.78


test loss
test loss 30.89, rec 27.83, adv 3.06, d0 0.73, d1 0.83


the outputs are saved in  ../../tmp/CrossAligned_VAE_z_merge
