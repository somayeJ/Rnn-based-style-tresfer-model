multi_decoder.py: 
	style transfer_multi decoder model
multi_decoder_condition.py:
	style transfer_multi decoder model, with the following condition:
	            if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimize0 = model.optimize_tot0
                        optimize1 = model.optimize_tot1
                    elif loss_d0<1.2 :
                        optimize0 = model.optimize_tot0
                        optimize1 = model.optimize_rec1
                    elif loss_d1<1.2:
                        optimize1 = model.optimize_tot1
                        optimize0 = model.optimize_rec0
                    else:
                        optimize0 = model.optimize_rec0
                        optimize1 = model.optimize_rec1
multi_decoder_condition_2.py, compared to multi_decoder_condition.py, it removes style parts and also back propagates the adv_loss only on generator only

In the paper, style_transfer_multi_decoder.py with 40 epochs is used and with no condition (based on the conditions of the original code)


style_transfer_multi_decoder emphasize_z_merge.py, zs merged to the outputs of the decoder and it is based on multi_decoder.py (and 40 epochs)
