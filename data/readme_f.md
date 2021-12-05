yelp: the original files of Yelp datase from the paper "cross aligned  style transfer"
yelp2: from the paper "https://github.com/fastnlp/style-transformer"

yelp_elmo_reps:
	yelp_elmo_rep_seqs: the elmo rep of sequences of yelp files, which is computed by the concatenation of the [average, min , max ] of embeddings of the tokens of each sequence with the shape 1024, across the sotoons, so the seq emb.shape of each sequence is 3072
	test.py: to read the seqs


