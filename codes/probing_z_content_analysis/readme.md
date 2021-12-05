-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
A) Descriptions of the folders 
1. data: 
Some data which is produced by me to investigate the performance of the models, the files in this folder with 0 and 1 suffix to create the embeddings of the two files with - and + style, at the same time (which is done in the new z_create files)
	yelp_test_paraphrase. : more similar than yelp_test_ori_paraphrase.
	yelp_test_ori_paraphrase.
Look at the readme.txt file in the folder of the data , so that u see the description of the files 

2. data_z_rep_of_models: emb reps of files in data folder created by different models and saved in the sub-folder named based on the that model

3. cosine_distance_computing:
The code, cosine_distance_sentences.py, in this folder is used for calculating cosine distance between embs of seqs:
	in cosine_distance_sentences.py, we insert the file names we want to compute the distance in between their embeddings as r_file and q_file in gen_score function
	To run the cosine_distance_sentences.py, we run the following in terminal: python cosine_distance_sentences.py  ../output_emb_files/(file directory in which r_file and q_file are stored) 

6. Output_emb_files: 
The folder contains the files which contain emb_seqs for the files in the data folder based on the models which the files are named for, for instance, pretrained_styleTransfer stands for the files which produced based on the style transfer pretrained model. Thses models are saved in the ../../tmp/

-------------------------------------------------------------------------------------------------------------------------------------------------------------
B)To do the probing z experiments:
B.1)Create emb_rep of seq:
3. To create z find (model_name)_z_create.py 
	1. Go to the folder where the codes of the considered model is saved (it is saved in the corresponding model code folder as (model_name)_z_create.py)
	2. if you can not find a (model_name)_z_create.py and want to create a (model_name)_z_create.py for a new model, 
		2.1 go to .../code-v-2/probing experiments folder and replace the class part of the new model with the class part pf the.../code-v-2/probing experiments.py/adv_SD_z_create.py 
		2.2 make sure that --keep_data_order boolean with default false is in the list of arguments (it should be set to true)
		2.3 make sure that get batches method of  the util(tf_v_1).py file  is the same as get batches method of the util(tf_v_1).py file of the probing classification_experiments folder 
		2.4 add args as the last argument of the method get_batches in (model_name)_z_create.py
		2.5. run this in terminal:
dv_SDS_z_create.py  --model ../../../../../../tmp/Deterministic_CrossAligned_models/singl│
e_encoder_single_decoder\(SDS\)/GYAFC_corpus/adv_z_init/model --vocab ../../../../../../tmp/Determinis│
tic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/GYAFC_corpus/adv_z_init/yelp.vocab   --lo│
ad_model true --output ../../../../../probing_z_content_analysis/data_z_rep_of_models/adv_SDS/GYAFC_mo│
dels/gyafc  --test ../../../../../probing_z_content_analysis/data/gyafc --keep_data_order true        │
                

arguments: --test shows where to read data from, we just write the file name with no .0 or .1 referring to the style , --keep_data_order: indicates keeping the data order and not reorder them based on the length, --model and --vocab show where the model and its vocab are saved,  --output shows where to save data, again we just write the file name with no .0 or .1 referring to the style 

B.2)computing cosine distance



---------------------------------------------------------------------------------------------------------------------------------------------------------
For the EACL conference, since  the files here are sentiment related , we add the following files:

GYAFC0: 20 randomly selected sequences from GYAFC test set (fisrt 10 from Music, Entertainment & second 10 F&R)
GYAFC0_para: the paraphrases of these seqs taken from the sets of the paraphrases with opposite style ref0 && ref1

GYAFC1: 20 randomly selected sequences from GYAFC test set
GYAFC1_para: the paraphrases of these seqs taken from the sets of the paraphrases with opposite style ref0 && ref1

1: formal && 0:	informal


**********: Looking at the paraphrase files, it seems like from formal to there are mre variations in the human-written files than informal to formal















