The folder "../opinion-lexicon-English" is the list of negative positive words taken from paper style transfer in text, style transfer in text exploration and evaluation!
the base code is code emb_test.py which is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text_style_transfer

************************************************************************************************************************************************************************************************************
Calcualting the embedding of the tokens, based on elmo model: 

	Code: 
	- StyleMarkerRemover.py is imported from inside the code elmo_seqEmb_similarity.py, for removing the style markers
	- Elmo_token_rep_write.py : computing and writing the elmo reps of the files in the given direcrory_write
	
____________________________________________________________________________________________________________________________________________________________________________________________________________


In file elmo_seqEmb_similarity.py,

		a. It has 2 below cases:
1. compute the content preservation between original files and reconstructed files OR original files and tsf files:
set the variable, remove_style_markers= False, else if you want to compute the content preservation between style shifted files and their corresponding origional files, set the variable, remove_style_markers= True, u should first UNCOMMENT the variable, remove_style_markers and also the block at the end of code and  COMMENT the block with for loop (remove_style_markers in [False,True])

2.1. compute the content preservation between original files and reconstructed files AND original files and tsf files:
u should first COMMENT the variable, remove_style_markers and also the block at the end of code and UNCOMMENT the block with for loop (remove_style_markers in [False,True])

		b. determine the pass to the original filesand target files, both text and elmo reps, in the following variables, respectively:
			directory_orig =  "../../../../data/yelp/" #  orig files
			directory_orig_elmo =  "../../../../data/yelp_elmo_rep/" # elm_rep of orig files

			directory_tgt = "../../../../tmp/CrossAligned_AE_emb/" # trg files
			directory_tgt_elmo = "../../../../tmp/CrossAligned_AE_emb/elmo_rep_test_gen_files/" #elm_rep of trg files


__________________________________________________________________________________________________________________________________________________________________________________________________________

If after running the file , u get the message that "There is no elm_rep of orig files in this directory and You should generate the elm_rep of trg files  using Elmo_token_rep_write.py", do so :

in file Elmo_token_rep_write.py:
	a.determine the pass to the directories of the original files, generated files and the directory that we want to write the elmo_tokenrep of the generated files , respectively:
		directory = "../../../data/yelp/sentiment.test" # yelp_dataset
		directory_read = "../../../tmp/CrossAligned_AE_emb/sentiment.test"
		directory_write = "../../../tmp/CrossAligned_AE_emb/elmo_rep_gen_files/sentiment.test.elmo."

	b. determine whether u want to calculate the emb of the seq or emb of the tokens of the sequences, by setting the value of the argument seq_embedding=True in create_elmo_embeddings() or False respectively (there is an error in case of calculating the embeddings for tokens)



	





