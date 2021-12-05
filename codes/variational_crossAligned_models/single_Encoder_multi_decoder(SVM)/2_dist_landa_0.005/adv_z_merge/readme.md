1. The files in this folder are based (copied) from files in sister folder "style_transfer_VAE" and just z (without label is merged with outputs of decpder at each time step)
2. Then the changes were made based on style_transfer_multi_decoder_AE_merge_z


The number of epochs are 20  (since it is 20 in style_transfer_VAE folder as well) and the condition for multi decoder is the same as original code (the same as what is reported in paper for style transfer muti-decoder frameworks)


the outputs are saved in tmp/multi_decoder_VAE_models/style_transfer_VAE_z_merge
