to investigate the effect of length of the orig and generated phrases on each of the aspects of the evaluation

1. content preservation: content_preservation.py to run it, run the follwing line in the terminal:
 	python content_preservation.py  ../models_outputs/AE/ ../models_outputs/AE/sentiment.test.  ../../../data/yelp/sentiment.test. '.rec'
	python content_preservation.py arg1: path to the file of semantic resemblance between phrases and generated files :../models_outputs/AE/
				       arg2: path to the generated file :../models_outputs/AE/sentiment.test.
				       arg3: :path to the original file :../../../data/yelp/sentiment.test.
				       arg-4: generated file_ suffix :'.rec'

	
			


