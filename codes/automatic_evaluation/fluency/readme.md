This code is for investigating the fluency of the sentences and their syntactical correctness which taken from the website https://github.com/shentianxiao/language-style-transfer/blob/master/code/language_model.py (paper style transfer with cross-alignment).

	1. models_yelp_data: "train0_model" and "train0+train1_model" are folders in which trained LMs are saved where yelp data is used
	2. models_GYAFC_data:
		 "train0+train1_model" are folders in which trained LMs are saved where GYAFC data is used, test pplx is 
		 "train0+train1_test0+test1_model is used for training
	3. models_GYAFC_data_yelp_data: 
		"train0+train1_model" are folders in which trained LMs are saved where GYAFC+yelp data is used, test pplx is 

For calcualting the pplx of files, do the following steps:
Do either A or B:
	A)
	1. determine the suffix of the file (in case of any, like tsf or rec) in the main function of the code, language_model.py 
	2.  Run the language_model_v1.py code in the terminal, by determining the arguments, like the example below:
		--test: path to the files we want to compute their pplx
		--vocab: path to the vocab of the trained model
		--model: path to the trained model
		--load_model true
		--shuffle_sentences: true, if we want to calcualte the fluency of a sentences where their tokens are shuffled, here thesuffix should be depending on the file we are calculating its fluency
		--higher_bound true (if we want to calculate the fluency of a synthectic file as a higher bound, where the fluency is supposed to be good) and false (if we want to calculate from pre exited files and suffix= 'tsf' or ... depending on suffix of the file that we want to generate its fluency),  default=False

	 python language_model_v1.py  --test ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/fullzsource_attention_bidirectional_gru_v3/sentiment.test --vocab ./models_yelp_data/train0+train1_model/yelp.vocab  --model ./models_yelp_data/train0+train1_model/model --load_model true	
	B)
	Run the language_model_v2.py code in the terminal, by determining the arguments, like the example below:
		--model_type  'transformer'  or 'seq2seq'
		--mode 'rec' or 'tsf'
		--test: path to the files we want to compute their pplx
		--vocab: path to the vocab of the trained model
		--model: path to the trained model
		--load_model true
		--shuffle_sentences: true, if we want to calcualte the fluency of a sentences where their tokens are shuffled, here thesuffix should be depending on the file we are calculating its fluency
		--higher_bound true (if we want to calculate the fluency of a synthectic file as a higher bound, where the fluency is supposed to be good) and false (if we want to calculate from pre exited files and suffix= 'tsf' or ... depending on suffix of the file that we want to generate its fluency),  default=False
		

	 python language_model_v2.py  --test ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/fullzsource_attention_bidirectional_gru_v3/ --vocab ./models_yelp_data/train0+train1_model/yelp.vocab  --model ./models_yelp_data/train0+train1_model/model --load_model true --model_type  'transformer'  --mode 'rec'	




For training the LMs, do the following steps:
	1. in the main function of the code language_model.py, determine the suffix of the files ('' in case of training and dev) ('tsf' or 'rec' in case of generated files ) 
	2. Run the code in the terminal, by determining the arguments, like the example below:
		--train: path to the files we want to train the LM on
		--dev: path to the files we want to use as dev while training
		--vocab: path to where we wanna save the vocab 
		--model: path to where we wanna save the model


	python language_model.py  --train ../../../data/yelp_data/sentiment.train --dev ../../../data/yelp_data/sentiment.dev --vocab ./models_yelp_data/train0+train1_model/yelp.vocab  --model ./models_yelp_data/train0+train1_model/model 






