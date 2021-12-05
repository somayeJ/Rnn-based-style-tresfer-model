*** To make the data of the two datasets of the same size to investigate its effect on the test losses, compared to upsampled model, to do so there is a need to assign the argument downsampling_train_SDS= True in terminal while training and test

To train and test with this model:
	***** 1. check the following arguments in the options.py files
		rec_factor
		max_epochs
		rev_tokens:  set the argument to True: if while training, you want the input seq to get reversed, or False: otherwise
		shuffle_tokens:  set the argument to True: if while training, you want the input tokens to get shuffled, or False: if u want the order of tokens to be saved
		word_drop: in case you do not want to drop words from the input while training, set it to 0.0, otherwise, set it to a value between (0 , 1)



