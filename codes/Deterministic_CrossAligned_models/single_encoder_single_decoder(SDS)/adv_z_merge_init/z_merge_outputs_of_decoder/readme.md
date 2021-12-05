For the strategy of merging z to the outputs, we tried the followings

1. z_merge: in which just z is merged with the outputs of the decoder at each state
	-outputs: saved in the folder z_merge in tmp folder
	-codes which are changed for in this approach: style_transfer_merge_z.py, beam_search_tf_1.py and utils_tf_1.py

2. labels+z_merge:in which the concatenation of the vectors desired_labels and z is merged with the outputs of the decoder at each time state
	-outputs: saved in the folder labels+z_merge in tmp folder
	-codes which are changed for in this approach: style_transfer_merge_labels_z.py, beam_search_tf_1_labels_z.py
	
3. labels+z_merge_non_rev_x :in which the concatenation of the vectors desired_labels and z is merged with the outputs of the decoder at each time state, and the input sequences are not reversed
	-outputs: saved in the folder labels+z_merge_non_rev_x in tmp folder
	-codes which are changed for in this approach: style_transfer_merge_labels_z_non_rev_x.py, beam_search_tf_1_labels_z.py, utils_tf_1_labels_z_non_rev_x.py




In the paper, we used the results of z_merge, adv_SDS_merge_z.py



dev loss 31.30, rec 28.06, adv 3.24, d0 0.66, d1 0.78
