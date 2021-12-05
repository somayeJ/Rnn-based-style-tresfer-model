In this folder, we do the content presevation by two techniques: 1. embedding based methods, 2. N-gram based methods:

	- The folder "opinion-lexicon-English" is the list of negative positive words taken from paper style transfer in text, style transfer in text exploration and evaluation! which is used in both    models

	- the base code, code emb_test.py, is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text_style_transfer

	- Elmo_embeddings folder: You can generate Elmo embeddings of seqs  (The file to generate elmo-embeddings is in ../automatic_evaluation/content_preservation/Elmo_embedding/Elmo_token_rep_write.py) 		and also compute the similaruty between the sequences using elmo_seqEmb_similarity.py

	- word_overlap.py for calculating the N-gram based methods

*************************************************************************************************************************************************************************************************************
1. embedding based methods: Calcualting the similarity between two sequences by computing embedding of the tokens, 
	1.1 based on the pre-trained glove embeddings 
	1.2 based on the pre-trained elmo embeddings (we have not done it throughly, we tried as a test on one model )

	Codes:
	-Embedding.py, Embedding_gensim.py, are imported  inside the files emb_test*.py and the path to embedding files are given there
	-Tools.py is imported from inside the files emb_test*.py 	
__________________________________________________________________________________________________________________________________________________________________________________

A. computing content preserve of reconstructed files,

-  use emb_test.py since there is no need for removing style markers :
	1- if you want to compute random content preservation which is between some random files (like amazon files) and generated files:
		a. set the variable, random_content_preservation to True
		b. set the value of suffix of the function gen_score_random() in the main function, to ".rec" or ".tsf", based on the generated file format in the main function (here .rec)
		c. run this in terminal emb_test.py path_generated_file, path random file: like the following example:  python emb_test.py ../eval/model_outputs/AE/sentiment.test.  ../../../data/amazon/binary/sentiment.train.

	2- if you want to compute the content preservation between generated files and their corresponding origional files:
		a. set the variable, random_content_preservation to False
		b. set the value of suffix of the function gen_score() in the main function, to ".rec" or ".tsf", based on the generated file format in the main function (here .rec)
		c. copy and paste origional files in the same folder as the generated files
		d. run this in terminal: python emb_test.py path_generated_file: like the following example:  python emb_test.py ../eval/model_outputs/AE

_______________________________________________________________________________________________________________________________________________________________________________

B. computing content preserve of style-shifted files,

- use emb_test_style_filtered_style_markers.py, since there is a need for removing style markers (comparing tsf files to some files (either 1. random or 2. original))

	1- Comparing generated style-shifted files to random files (like Amazon files or yelp training files (all results in the paper are as a result of using training yelp models)):
		a. set the variable, random_content_preservation = True, set the variable remove_style_markers = True
		b. set the value of suffix of the function gen_score_random() in the main function, to ".rec" or ".tsf", based on the generated file format in the main function (here .tsf)
		c. if you want to remove the style markers set, remove_style_markers to True other to False
		d. In terminal run: emb_test_style_filtered_style_markers.py path_generated_file, path random file: like the following example: python  emb_test_style_filtered_style_markers.py   ../../../tmp/CrossAligned_VAE_z_merge/sentiment.test.  ../../../data/yelp/sentiment.train.

	2- Comparing generated style-shifted files to their corresponding origional files:
		a. set the variable, random_content_preservation to False, 
		b. set the value of suffix of the function gen_score() in the main function, to ".rec" or ".tsf", based on the generated file format in the main function (here .tsf)
		c. if you want to remove the style markers set, remove_style_markers to True other to False
		d. run this in terminal: python emb_test_style_filtered_style_markers.py path_generated_file: like the following example:python  emb_test_style_filtered_style_markers.py ../eval/model_outputs/AE/  ../../../data/yelp/  

**: For calculating content preservation it makes no difference in practice to use part A or part B (if we use B there is no need for removing style markers, so we set  remove_style_markers = False) but for random content preservation, i received different results by running these two files, so we use this file (B))


***: In Aics paper, we keep the style markers
***: In coling paper, we remove the style markers
***: Lower bound reported in paper are as a result of computing distance between yelp/GYAFC/amazon trf files and  training files from yelp/GYAFC/amazon corpora while removing the style markers
***: Lower bounds reported in Transfer report, we as result of randomly sampling from train & test sets of the Yelp_coling and GYAFC preprocessed
**************************************************************************************************************************************************************************************************************
2. N-gram based methods: based on the method of word_overlap of the paper Disentangled Representation Learning for Non-Parallel Text Style Transfer, https://www.aclweb.org/anthology/P19-1041/
(the code is also based on the code of this paper which is in tf.1.*

(** In our calculations, we keep-stop-w & remove corresponding-style markers of the files, it is the opposite in the papers)
Files:
	1.***word_overlap.py
	2.***word_overlap_each_line.py : this file stores the word overlap of the seq pairs and stores them in files
	**arguments are:
		arg1: path to file1 (source file)
		arg2: path to file2 (generated file)
		arg3: style of file1
		arg4: style of file2
		arg5: path to the file that we write the scores of word overlap of each 2 seqs 
Run the following in the terminal:
 python word_overlap_each_line.py ../../../data/yelp/sentiment.test.0    ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/attention_z_init_bi_gru/test1/sentiment.test.0.tsf  ""  ""  ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/attention_bidirectional_gru_v3/word_overlap_sim_score_0.txt  


To compute the content preservation using this method, follow these steps:
	1. Use the code word_overlap.py
	2. If u want to keep the stop words in both files, in the function "word_overlap_score_evaluator" comment "english_stopwords = get_stopwords()" and uncomment "english_stopwords = set([])". If u want to remove the  stop words in both files, do the other way around.
	3.A. if you want to keep the style_markers: set "remove_style_markers = False" : how u set the args 3 and 4 makes no difference here
	3.B. if you want to remove the style_markers: set "remove_style_markers = True"
	
	3.B.1: If u want to remove both styles from each file regardless of its style (for example, if style of file1 is negative, we remove both negative and positive style markers), run this in the terminal: 
	python word_overlap.py ../../../data/yelp/sentiment.test.0  ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/Yelp_corpus/adv_z_init/sentiment.test.0.tsf  ""  ""
	
	OR 3.B.2: if u want to remove the style-markers with regards to the style of each file (for example, if style of file1 is negative, we just remove the negative style markers), run this in the terminal: 
	python word_overlap.py ../../../data/yelp/sentiment.test.0  ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/Yelp_corpus/adv_z_init/sentiment.test.0.tsf  "negative"  "positive"

**arguments are:
	arg1: path to file1 (source file)
	arg2: path to file2 (generated file)
	arg3: style of file1
	arg4: style of file2
	***: if you want to keep style markers, u can set arg3 and arg4 to "" & ""

** in the paper Disentangled Representation Learning for Non-Parallel Text Style Transfer, https://www.aclweb.org/anthology/P19-1041/, they removed stop words and also removed both styles from each file regardless of its style 
	in our results of the slides with single_eval_metric, we remove the corrsponding style marker of each file regarding its style, and keep SW 

*****:in AICs paper and EACL paper: we keep the style markers, and remove SW 


************************************************************************************************************************************************************************************************************
 3. Word mover distance :
	1. run this in terminal python  WMD.py arg1:path to file 1  arg2: path to file2 arg3: path and name of file3 which is csv
	file1& file2: files that we want to comapre
	file3: file which we want to write seqs whose wmd is inf
	2. we remove those seqs whose wmd is inf and compute the mean value for the rest
	3. While processing the sequences, we replace the unk with the, also we add the to the seqs whose lengths are 0

