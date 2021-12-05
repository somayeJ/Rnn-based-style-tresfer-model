"""
Word Movers' Distance
=====================
dowloaded from https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
Demonstrates using Gensim's implemenation of the WMD.

"""

###############################################################################
# Word Mover's Distance (WMD) is a promising new tool in machine learning that
# allows us to submit a query and return the most relevant documents. This
# tutorial introduces WMD and shows how you can compute the WMD distance
# between two documents using ``wmdistance``.
#
# WMD Basics
# ----------
#
# WMD enables us to assess the "distance" between two documents in a meaningful
# way, even when they have no words in common. It uses `word2vec
# <http://rare-technologies.com/word2vec-tutorial/>`_ [4] vector embeddings of
# words. It been shown to outperform many of the state-of-the-art methods in
# *k*\ -nearest neighbors classification [3].
#
# WMD is illustrated below for two very similar sentences (illustration taken
# from `Vlad Niculae's blog
# <http://vene.ro/blog/word-movers-distance-in-python.html>`_\ ). The sentences
# have no words in common, but by matching the relevant words, WMD is able to
# accurately measure the (dis)similarity between the two sentences. The method
# also uses the bag-of-words representation of the documents (simply put, the
# word's frequencies in the documents), noted as $d$ in the figure below. The
# intuition behind the method is that we find the minimum "traveling distance"
# between documents, in other words the most efficient way to "move" the
# distribution of document 1 to the distribution of document 2.
#

# Image from https://vene.ro/images/wmd-obama.png
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import sys
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import csv
#img = mpimg.imread('wmd-obama.png')
#imgplot = plt.imshow(img)
#plt.axis('off')
#plt.show()

###############################################################################
# This method was introduced in the article "From Word Embeddings To Document
# Distances" by Matt Kusner et al. (\ `link to PDF
# <http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf>`_\ ). It is inspired
# by the "Earth Mover's Distance", and employs a solver of the "transportation
# problem".
#
# In this tutorial, we will learn how to use Gensim's WMD functionality, which
# consists of the ``wmdistance`` method for distance computation, and the
# ``WmdSimilarity`` class for corpus based similarity queries.
#
# .. Important::
#    If you use Gensim's WMD functionality, please consider citing [1], [2] and [3].
#
# Computing the Word Mover's Distance
# -----------------------------------
#
# To use WMD, you need some existing word embeddings.
# You could train your own Word2Vec model, but that is beyond the scope of this tutorial
# (check out :ref:`sphx_glr_auto_examples_tutorials_run_word2vec.py` if you're interested).
# For this tutorial, we'll be using an existing Word2Vec model.
#
# Let's take some sentences to compute the distance between.
#

# Initialize logging.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'

###############################################################################
# These sentences have very similar content, and as such the WMD should be low.
# Before we compute the WMD, we want to remove stopwords ("the", "to", etc.),
# as these do not contribute a lot to the information in the sentences.
#

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
#download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    #return [w for w in sentence.lower().split() if w not in stop_words]
    #return [w for w in sentence.lower().split() if w not in stop_words]
    sen_tokens =sentence.lower().split() 
    preprocess_sen = []
    for w in sen_tokens:
    	if w in ['<unk>']:
    		preprocess_sen.append('the')
    	else:
    		preprocess_sen.append(w)
    if len(preprocess_sen)  == 0:
    	preprocess_sen.append('the')

    return preprocess_sen


#sentence_obama = preprocess(sentence_obama)
#sentence_president = preprocess(sentence_president)

###############################################################################
# Now, as mentioned earlier, we will be using some downloaded pre-trained
# embeddings. We load these into a Gensim Word2Vec model class.
#
# .. Important::
#   The embeddings we have chosen here require a lot of memory.
#
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
def wmd_seqs(model,seq1,seq2):
	###############################################################################
	# So let's compute WMD using the ``wmdistance`` method.
	#
	distance = model.wmdistance(preprocess(seq1),preprocess(seq2))
	print('distance = %.4f' % distance)
	return distance


def normal_wmd_seqs(model,seq1,seq2):
	###############################################################################
	# Normalizing word2vec vectors
	# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#
	# When using the ``wmdistance`` method, it is beneficial to normalize the
	# word2vec vectors first, so they all have equal length. To do this, simply
	# call ``model.init_sims(replace=True)`` and Gensim will take care of that for
	# you.
	#
	# Usually, one measures the distance between two word2vec vectors using the
	# cosine distance (see `cosine similarity
	# <https://en.wikipedia.org/wiki/Cosine_similarity>`_\ ), which measures the
	# angle between vectors. WMD, on the other hand, uses the Euclidean distance.
	# The Euclidean distance between two vectors might be large because their
	# lengths differ, but the cosine distance is small because the angle between
	# them is small; we can mitigate some of this by normalizing the vectors.
	#
	# .. Important::
	#   Note that normalizing the vectors can take some time, especially if you have
	#   a large vocabulary and/or large vectors.
	#
	model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.

	distance = model.wmdistance(preprocess(seq1), preprocess(seq2))
	print('distance = %.4f' % distance)
	return distance, np.mean(distance)

def normal_wmd_files(file1,file2,file3):
	distance = []
	inf_seqs = []
	with open(file1) as f1, open(file2) as f2, open(file3, 'w') as f3:
		
		for no,(line1, line2) in enumerate(zip(f1,f2)):
			print("shomare sequence",no, " seq no")
			d=normal_wmd_seqs(model,line1,line2)
			if np.isinf(d[0]):
				inf_seqs.append([no, line1,line2])

			else:
				distance.append(d)
				print(line1,line2) 
	  

		write = csv.writer(f3)
		write.writerows(inf_seqs) 


	return distance, np.mean(distance)
	
if __name__=="__main__":
	distance_normal,mean_distance = normal_wmd_files(sys.argv[1],sys.argv[2],sys.argv[3])
	print(distance_normal)
	print('**************************************')
	print('***************************************')
	print('***************************************')
	print('***************************************')
	print('distance_normal_mean = %.4f' % mean_distance)
	with open ('test_wmd.txt','w') as f:
		f.write(str(mean_distance))
'''
		reader = csv.reader(csvfile)
		for r in reader:
			print(r)




	


###############################################################################
# References
# ----------
#
# 1. Ofir Pele and Michael Werman, *A linear time histogram metric for improved SIFT matching*\ , 2008.
# 2. Ofir Pele and Michael Werman, *Fast and robust earth mover's distances*\ , 2009.
# 3. Matt Kusner et al. *From Embeddings To Document Distances*\ , 2015.
# 4. Thomas Mikolov et al. *Efficient Estimation of Word Representations in Vector Space*\ , 2013.
#



'''
