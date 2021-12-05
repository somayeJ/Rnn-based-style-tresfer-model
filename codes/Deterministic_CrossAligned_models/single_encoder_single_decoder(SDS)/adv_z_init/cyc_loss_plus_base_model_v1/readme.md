For training && testing the models:

***** 1. check the following arguments in the options.py files
	rec_factor
	cycl_factor
	max_epochs
	rev_tokens:  set the argument to  True: if while training, you want the input seq to get reversed, or False otherwise
	cyclic_soft: set the argument to  True: if while training, you want to compute the cyclic loss by using soft samples, or False: if you want to compute the cyclic loss using hard samples

***** 2. check the data tokens of the input are shuffled or no, by setting nosiy argument in get_batches() to True (in case of need for reordering tokens) or False (in case of need for keeping the order of  tokens) in adv_SDS.py function while loading the train data
