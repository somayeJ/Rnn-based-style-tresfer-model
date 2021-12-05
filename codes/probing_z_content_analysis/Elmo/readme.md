To create the Elmo embeddings of the tokens for each sequence in the files given, we need to give the directory of files in each of the codes Elmo_*.py and also we need to make the needed changes in the function load_all_dataset according to the names of the files we have.

To compare two given files considering their elmo reps for seqs , we need to use the code Elmo_keras_randomfiles.py  after doing the above changes in the code and more over, we give the name of the two files we want to compare their embeddings in line 147 of the code res = gen_score(amazon_1_emb,synthetic_none_sense_emb ) of Elmo_keras_randomfiles.py
These are the description of the codes we have:

	1. Elmo_keras.py (the origional code sent by Nazanin, run on IMDB dataset)

	2. Elmo_keras_modified_yelp.py (the file modified by me, run on Yelp dataset)

	3. Elmo_keras_randomfiles.py (the file modified by me, run to create the emb seq of the data we created for comparing models in terms of embeddings they create, here we calculate the embeddings of the sequences in these files using ELMO model to compare it with the performance of the models in terms of embeddings), the following two code_files are called inside Elmo_keras_randomfiles.py to compute the distance between the embedding seqs of the given files.

	4. Elmo_keras_write.py:me: modofied Elmo_keras.py for writing the elmo embeddings of files of yelp data in directory_write

	5. Embedding.py: the functions in this file are called by other codes is called by emb_test.py
	6. emb_test.py : the functions in this file are called by other codes is called by Elmo_keras_randomfiles.py 
	7. Tool.py: the functions in this file are called by other codes Elmo_*.py 


	
