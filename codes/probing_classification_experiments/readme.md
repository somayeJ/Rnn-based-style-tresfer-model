

1. Data folder (there are 3 folders in data folder section as follows):
**: in the reported results upsamling method has been implemented
		1. upsampling_emb:
 the emb representations of yelp data set is saved here produced as the last state of encoder in different models specified by the name of folders and the test, train and dev data are balanced by considering the length of bigger sized corpus and adding to the length of smaller sized corpus by replicating the data

		2. downsampling_emb:
 the emb representations pf yelp data set is saved here produced as the last state of encoder in different models specified by the name of folders and the train, test and dev data are balanced by considering the length of smaller sized corpus and considering the the same length cut of the bigger sized corpus, 

		3. origional_synthetic:
 just 8 sentences that I write to test the classifiers. Their embeddings are saved in ...probing classification_experiments/data/increased_data_balanced_emb/(model_name)/synthetic

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. Models folder:
	probing classifier trained with the emb created by different models are saved here either in upsampling or downsampling folder (depending on which method to use)

The models saved in each of the two folders can be one of the folowings:
	1.adv_SDS: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_SD,
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_SD/yelp
		create_embedding file .../code-v-2/probing_classification_experiments/data_emb/adv_SD_z_create.py
	2.adv_SVS: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_SV,
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_SV/yelp
		create_embedding file .../code-v-2/CrossAligned_VAE/adv_SV_z_create.py
	3.adv_MDS: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_MS
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_MD/yelp
		create_embedding file .../code-v-2/multi_decoder_models/deterministic_models/adv_SV_z_create.py
	4.adv_MVS:
	5.adv_MDS_merge_z: 
		model is saved in .../code-v-2/probing_classification_experiments/models/adv_MS_merge_z
		yelp data embedding rep is saved in .../code-v-2/probing_classification_experiments/data/emb/adv_MD_merge/yelp
		create_embedding file .../code-v-2/multi_decoder_models/deterministic_models/adv_SV_merge_z_create.py
	6.adv_SVS_merge_z: 
	7.adv_MDS_merge_z: 
	8.adv_SDS_merge_z:
	9. ...

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3. Codes
	1. adv_SDS_z_create.py is based on "style_transfer_sentence_emb_cross_aligned_models.py"
	It can be used to create z_embeddings of seqs (the codes in this folder are copied from .../language-style-transfer-barn/code-v-2/sequence_emb_z/create_emb_of_seqs_codes0 and for the framework adv_SDS.py).

	2. classifier_probing.py is based on the classifier on page https://github.com/shentianxiao/language-style-transfer/blob/master/code/style_transfer.py

__________________________________________________________________________________________________________________________________________________________________________________________________

To run classifier_probing.py, There are two steps(A&&B):

A) Preparing data: create z_embeddings of the test, train and dev data using the framework we want to do the probing task on:
	1. For creating embedding of different models, there is a need to create (model_name)_z_create.py file, to do so, 
		1. copy the sample_z_create_code.py file to the folder of the considered framework 
		2. copy the "class Model" and the  import section part above the "class Model" of the considered framework and paste on top of the sample_z_create_code.py file uptill the method seq_embeddings
		3. rename sample_z_create_code.py to (considered_framework_name)_z_create.py

	2.To create emb using method downsampling_emb, there is a need to run (considered_framework_name)_z_create.py, to create this file, follow these steps:
		1. add the  followings in options.py   
	argparser.add_argument('--downsampling', 
		    type=bool,
		    default=False)
	argparser.add_argument('--keep_data_order',
            type=bool,
            default=False)
		2. copy the methods of get_batch & get_batches of the file "utils.py" in the folder ../codes/probing_classification_experiments and paste it instead of the methods of get_batch & get_batches in utils.py or utils_tf_1.py (consider the one which imported in (model_name)_z_create.py) of the folder of the model where (model_name)_z_create.py is 

		3. add the argument "args" in get_batches method (model_name).py and all other model files which import utils.py or utils_tf_1.py in which  we did the last step change 

		4. make sure that --keep_data_order argument (which is for comparing file seqs) is set to False

	3. To create z_embeddings of the data, go to the folder where (model_name)_z_create.py is placed, create the embeddings by using one of the methods below:
		3.A. Downsampling method to create emb of data: 
			set --downsampling to True & Run (model_name)_z_create.py in terminal, like the following:
			python sample_z_create_code.py --downsampling true --dev ../../data/yelp/sentiment.dev  --load_model true --model ../../tmp/CrossAligned_AE_z_emb/model --vocab ../../tmp/CrossAligned_AE_z_emb/yelp.vocab  --output ./data/downsampling_models/adv_SD/sentiment.dev
	
		3.B. Upsampling method to create emb of data:
			set --downsampling to False & run (model_name)_z_create.py in terminal, by following:
 python adv_SDS_z_create.py  --dev ../../../../../../data/amazon/binary/sentiment.dev   --load_model true --model ../../../../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/amazon_corpus/adv_z_init/model --vocab ../../../../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/amazon_corpus/adv_z_init/yelp.vocab  --output ../../../../../probing_classification_experiments/data/upsampling_emb/adv_SDS/amazon/sentiment.dev
                                                                                  

B) Train and test the classifier: Go to ../codes/probing_classification_experiments folder
	1. Training the classifier (it does train and test at the same time): 
	Give the arguments of train, test and dev (where to read data which is embedding reps of textual seqs, so of size [data_size, emb_size]) and model (where to save the model).
	run this in terminal for training classifier
	python classifier_probing.py --max_epochs 20 --train ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.train  --dev ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.dev  --test ./data/upsampling_emb/adv_MV_1_dist_z_merge/sentiment.test --model ./models/upsampling_models/adv_MV_1dist_z_merge/model
	2. Testing the classifier (loading the model to test): 
	Give the arguments test (where to read data which is embedding reps of textual seqs, so of size [data_size, emb_size]) and where to load the trained model + --load_model true

_________________________________________________________________________________________________________________________________________________________________________________________________


