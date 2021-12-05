1. Data folder (there are 3 folders in data folder section as follows):
		1. upsampling_emb:
 the emb representations of yelp data set is saved here produced as the last state of encoder in different models specified by the name of folders and the test, train and dev data are balanced by considering the length of bigger sized corpus and adding to the length of smaller sized corpus by replicating the data

		2. downsampling_emb:
 the emb representations pf yelp data set is saved here produced as the last state of encoder in different models specified by the name of folders and the train, test and dev data are balanced by considering the length of smaller sized corpus and considering the the same length cut of the bigger sized corpus, 

		3. origional_synthetic:
 just 8 sentences that I write to test the classifiers. Their embeddings are saved in ...probing classification_experiments/data/increased_data_balanced_emb/(model_name)/synthetic

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. models folder:
	probing classifier trained with the emb created by different models are saved here either in upsampling or downsampling folder (depending on which method to use)

The models saved in each of the two folders can be one of the folowings:
	1.adv_SD: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_SD,
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_SD/yelp
		create_embedding file .../code-v-2/probing_classification_experiments/data_emb/adv_SD_z_create.py
	2.adv_SV: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_SV,
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_SV/yelp
		create_embedding file .../code-v-2/CrossAligned_VAE/adv_SV_z_create.py
	3.adv_MD: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_MS
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_MD/yelp
		create_embedding file .../code-v-2/multi_decoder_models/deterministic_models/adv_SV_z_create.py
	4.adv_MV:
	5.adv_MD_merge_z: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_MS_merge_z
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_MD_merge/yelp
		create_embedding file .../code-v-2/multi_decoder_models/deterministic_models/adv_SV_merge_z_create.py
	6.adv_SV_merge_z: 
	7.adv_MD_merge_z: 
	8.adv_SD_merge_z:
	9. ...

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. codes
	1. adv_SDS_z_create.py is based on "style_transfer_sentence_emb_cross_aligned_models.py"
	It can be used to create z_embeddings of seqs (the codes in this folder are copied from .../language-style-transfer-barn/code-v-2/sequence_emb_z/create_emb_of_seqs_codes0 and for the framework adv_SDS.py).

	2. classifier_probing.py is based on the classifier on page https://github.com/shentianxiao/language-style-transfer/blob/master/code/style_transfer.py

__________________________________________________________________________________________________________________________________________________________________________________________________

To run classifier_probing.py, There are two steps:

A) creating z_embeddings of the test, train and dev data using the framework we want to do the probing task on:
	1. For creating embedding of different models, there is a need to create (model_name)_z_create.py file, to do so, we copy the file sample_z_create_code.py and paste it in the folder of that framework, then we copy  the class Model and import section of that framework  and paste it in the sample_z_create_code.py before def seq_embeddings functionfile and rename this file to (model_name)_z_create.py file
	To create emb using method downsampling_emb, there is a need to do the following (model_name)_z_create.py, as follow:
		1. add the  following in options.py   argparser.add_argument('--downsampling', 
		    type=bool,
		    default=False)
		2. add args in get_batches method in (model_name)_z_create.py
		3. copy  get_batches method in utils.py of .../probing_classification_experiments folder (method copied here) and paste it instead of  get_batches method in utils.py or utils_tf_1.py 			of the folder of the model where (model_name)_z_create.py is
		4. make sure that --keep_data_order argument (which is for comapring file seqs) is set to False
		To Run (model_name)_z_create.py using downsampled_emb, Run the following in terminal:
python adv_SD_z_create.py --downsampling true --dev ../../data/yelp/sentiment.dev  --load_model true --model ../../tmp/CrossAligned_AE_z_emb/model --vocab ../../tmp/CrossAligned_AE_z_emb/yelp.vocab  --output ./data/downsampling_models/adv_SD/sentiment.dev

	
	2. To create emb using method upsampling_emb, set --downsampling to False
	To Run (model_name)_z_create.py using upsampled_balanced_emb, Run the following in terminal:
python adv_SD_z_create.py  --embedding  ../../word_emb/glove.6B.100d.txt --dev ../../data/yelp/sentiment.dev   --load_model true --model ../../tmp/CrossAligned_AE_emb/model --vocab ../../tmp/CrossAligned_AE_emb/yelp.vocab  --output ./data/upsampling_models/adv_SD/sentiment.dev

B) Train and test the classifier: Go to ../codes/probing_classification_experiments folder
	1. Training (it trains and tests at the same time): Give the arguments of train, test and dev (where to read data which is embedding reps of textual seqs, so of size [data_size, emb_size]) and model (where to save the model).
	run this in terminal for training classifier
	python classifier_probing.py --max_epochs 20 --train ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.train  --dev ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.dev  --test ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.test --model ./models/upsampling_models/adv_MV_1dist_z_merge/model
	2. Test: Give the arguments test (where to read data which is embedding reps of textual seqs, so of size [data_size, emb_size]) and where to load the trained model 

_________________________________________________________________________________________________________________________________________________________________________________________________


