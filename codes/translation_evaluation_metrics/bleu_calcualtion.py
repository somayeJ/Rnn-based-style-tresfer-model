# me
'''
The BLEU score consists of two parts, modified precision and brevity penalty. Details can be seen in the paper. 
You can use the nltk.align.bleu_score module inside the NLTK. One code example can be seen as below

Note that the default BLEU score uses n=4 which includes unigrams to 4 grams. If your sentence is smaller
than 4, you need to reset the N value, otherwise ZeroDivisionError: Fraction(0, 0) error will be returned. 
So, you should reset the weight.
''' 
import nltk

file_name_ref = "../sequence_emb/data/synthetic_none_sense.txt"
file_name_gen = "../sequence_emb/data/amazon_1.txt"

def read_data(file_name_ref, file_name_gen):
	with open(file_name_ref) as ref, open(file_name_gen) as gen:
		
		gens = [line.strip().split() for line in gen.readlines() ]
		refs = [line.strip().split() for line in ref.readlines()]
	refs_list = [[line] for line in refs]
	return refs_list, gens

'''

hypothesis = ["open", "the", "file"]
reference = ["open", "file"]
#the maximum is bigram, so assign the weight into 2 half.
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = (0.5, 0.5))
print BLEUscore
'''

#hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
#reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references


# it is for calculating the blue score of two sentences
references , generated_seqs = read_data(file_name_ref, file_name_gen) 
blue_list = []
#BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], genrated_seq)
for (reference,generated_seq) in zip(references, generated_seqs):
	BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, generated_seq)
	blue_list.append(BLEUscore)
print ('blue_list',blue_list)

# computing corpus BLEU
# When it comes to corpus_bleu() list_of_references parameter, it's basically a list of whatever the sentence_bleu() takes as references
'''
def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None):
    """
    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :return: The corpus-level BLEU score.
    :rtype: float
    """
'''

#print(type(list_of_references), type(list_of_references))

corpus_bleu1 = nltk.translate.bleu_score.corpus_bleu(references, generated_seqs, weights=(1.0,0,0,0)) # unigram
corpus_bleu2 = nltk.translate.bleu_score.corpus_bleu(references, generated_seqs, weights=(0.5,0.5)) # bigram
'''
example:
r=[[["hello ","I ","am", "really", "happy"]]]
c= [["hello ","I ","am", "happy"]]
corpus_bleu1 = nltk.translate.bleu_score.corpus_bleu(r, c, weights=(1.0,0,0,0))
'''
print('corpus_bleu1', corpus_bleu1,"corpus_bleu2s", corpus_bleu2)