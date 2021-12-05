
codes from_action: Elmo folder is copied from action server  to create Elmo embeddings of the seqs

codes_for creating_elmo_data: I wrote the files in this folder to create elmo embeddings for tokens of the seqs of a given file: 
	to create the results:
	1. activate the  virtual environment venv_tf_2
	2. run  the following code in the terminal:  python main.py 
	--input_dir ../../../../data/yelp2/sentiment.test # directory + file_name (without style) where data is read
	--output_dir ../../../../data/yelp2_elmo/test # # directory + file_name (without style) where elmo_embeddings should be saved 
	--style_1 '.0'
	--style_2 '.1'
***** if the file does not have an style set --style_1 & --style_2 to ''
***** the output contains each token in a line

	

	
********************************************************************************************************************************************************************************************************************
Old readme about files of the folder codes from_action


	To create the Elmo embeddings of the tokens for each sequence in the files given, we need to give the directory of files in each of the codes Elmo_*.py and also we need to make the needed changes in the function load_all_dataset according to the names of the files we have.

	To compare two given files considering their elmo reps for seqs , we need to use the code Elmo_keras_randomfiles.py  after doing the above changes in the code and more over, we give the name of the two files we want to compare their embeddings in line 147 of the code res = gen_score(amazon_1_emb,synthetic_none_sense_emb ) of Elmo_keras_randomfiles.py
	These are the description of the codes we have:

		1. Elmo_keras.py (the origional code sent by Nazanin, run on IMDB dataset)

		2. Elmo_keras_modified_yelp.py (the file modified by me, run on Yelp dataset)

		3. Elmo_keras_randomfiles.py (the file modified by me, run to create the emb seq of the data we created for comparing models in terms of embeddings they create, here we calculate the embeddings of the sequences in these files using ELMO model to compare it with the performance of the models in terms of embeddings), the following two code_files are called inside Elmo_keras_randomfiles.py to compute the distance between the embedding seqs of the given files.

		4. Embedding.py: the functions in this file are called by other codes (by emb_test.py)
		5. emb_test.py : the functions in this file are called by other codes (Elmo_keras_randomfiles.py)
		6. Tool.py: the functions in this file are called by other codes (Elmo_*.py )


	
