The codes is for testing style shift power and is from the paper "Style Transfer from Non-Parallel Text by Cross-Alignment".

	1.yelp_models: the folders "model_balanced_data" and "model_non_balanced_data" are the trained models for the classifier.
	2.GYAFC_models: the folder "model_non_balanced_data" has the trained model for the classifier.
	3.amazon_models: the folder "model_non_balanced_data" has the trained model for the classifier.

	2. to test the style of the files, we follow these steps do either 1 or 2:
		1.  to run the code classifier_v1.py the following steps should be done: 
				a. in classifier_v1.py, we need to put the suffix rec or tsf or "" in the method, prepare(), depending on our data
				b. run the code in the terminal by specifying the arguments, like the example below:
			 	python classifier_v1.py --test ../../../tmp/multi_decoder_models/AE_multi_decoder/AE_multi_decoder_40_epochs/sentiment.test   --vocab ./model_balanced_data/yelp/yelp.vocab --model ./model_balanced_data/yelp/model --load_model true

		2.  to run the code classifier_v2.py the following command should be done: 
		run the code in the terminal by specifying the arguments, like the example below:
			 	 python classifier_v2.py  --test ../trsnsfrmr_outputs/A2/ --model_type  'transformer'  --mode 'rec'  --vocab ./yelp_trnsfrmr_paper/yelp.vocab --model ./yelp_trnsfrmr_paper/model  --load_model true


			--test: shows the path to generated files that we want to see the strength of the their style shift (tsf files) or style preserve (rec files)
		 	--vocab: path to where the vocab of the trained model, 
			--model: path to trained model of the classifier 
			--load_model true

	3. to train the classifier,  we follow these steps:
		1.  to run the code the following command should be run in the terminal: 
				a. in the classifier.py, we need to put the suffix rec or tsf or "" in the method, prepare(), depending on our data
				b. run the code in the terminal by specifying the arguments, like the example below:
		python classifier.py --train ../../../data/.../sentiment.train --dev ../../../data/.../sentiment.dev  --vocab ./model_balanced_data/yelp/yelp.vocab --model ./model_balanced_data/yelp/	
		
******************************************************

In the paper, we are reporting the values using non-balanced model


	





