0: paper

1: news

The files paper.txt & news.txt are the original files.

The first split contains files with no preprocessing steps, 
 

The sentiment.*.* in this file are the data partions pf the paper.txt & news.txt after the preprocessing steps of 
	- lowercasing
	- replacing digits with _num_ (like yelp data)
	- removing sentences with the length more than 20, 1327 were removed from from paper.text which 1.2% of the whole data
the size of dev & test are 2000, they are randomly selected from the  paper.txt & news.txt

We followed the paper (Style Transfer in Text: Exploration and Evaluation) which introduced this dataset for the partioning sizes and preprocessing steps.
