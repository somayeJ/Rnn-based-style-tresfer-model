in the folder "adding_two_datasets" the data of the two domains of Entertainment_Music & Family_Relationships is merged to create a dataset of formality / informality

0: informal
1: formal

data_origional: produced as a result of running the code "file_append.py", 

data_model: it replaces the dev and test sets to be more consistent with the size of data in Yelp dataset, since in yelp the size of test data is bigger than dev data and that is the data we use for training the models



Composing data: appending the files of style 1 or 0 for each of the test , dev and train partitions from E&M and F&R to form their corresponding datasets

Preprocessing steps (using the file: ./preprocess_data.py):

	1. removing seqs with len bigger than 30 && smaller than 4, we  chose these lens to have outliers around 1% and avoid removing  many seqs
	2. lowercasing
	3. replacing the numbers with _num_


the preprocessed files are saved here: '.sftp://jafarita@barn-e-03/vrac/jafari/workplace/shared-folder/style_transfer/language-style-transfer-barn/data/GYAFC_Corpus/adding_two_datasets/data_model/data_preprocessed/'


the files before processing are saved here: '.sftp://jafarita@barn-e-03/vrac/jafari/workplace/shared-folder/style_transfer/language-style-transfer-barn/data/GYAFC_Corpus/adding_two_datasets/data_model/data_before_preprocessed/'

***************** For the ICANN paper,  we used the processed GYAFC



