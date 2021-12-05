#run in the terminal: python data_len_analysis.py
import statistics


def len_analysis(filename):
	with open(filename) as f:
		lines0=f.readlines()

		t=0
		o=0
		e =0
		m=0
		for line in lines0:
			if len(line.split('\t'))==3:
				t+=1
			elif len(line.split('\t'))==1:
				o+=1
			elif len(line.split('\t'))>3:
				m+=1
			else:
				e+=1
		print('three',t,'ones',o,'more than 3',m,'else',e)
		lines=[]
		for line in lines0:
			if len(line.split('\t'))>1:
				lines.append(line.split('\t')[2])
			else:
				lines.append(line.split('\t')[0])
		len_sent = [len(line.split()) for line in lines ]
		max_len= max(len_sent)
		for line in lines :
			if len(line.split()) ==3:
				print(len(line.split()))
				print(line)
		print('max', max(len_sent))
		print('mean', sum(len_sent)/len(len_sent))
		print('median', statistics.median(len_sent))
		zero = [1 for line in lines if len(line.split())<1]
		one = [1 for line in lines if 0<len(line.split())<2]
		two = [1 for line in lines if 0<len(line.split())<3]
		three = [1 for line in lines if 0<len(line.split())<4]
		four = [1 for line in lines if 0<len(line.split())<5]
		five = [1 for line in lines if 0<len(line.split())<6]
		ten = [1 for line in lines if 0<len(line.split())<11]
		twenty = [1 for line in lines if 10<len(line.split())<21]
		thirty = [1 for line in lines if 20<len(line.split())<31]
		forty = [1 for line in lines if 30<len(line.split())<41]
		fifty = [1 for line in lines if 40<len(line.split())<51]
		sixty = [1 for line in lines if 50<len(line.split())<61]
		sixtyFive = [1 for line in lines if 60<len(line.split())<66]
		seventy = [1 for line in lines if 65<len(line.split())<71]
		seventyFive = [1 for line in lines if 70<len(line.split())<76]
		eighty = [1 for line in lines if 75<len(line.split())<81]
		nighty = [1 for line in lines if 80<len(line.split())<91]
		hundred = [1 for line in lines if 90<len(line.split())<101]
		hundredfifty = [1 for line in lines if 100<len(line.split())<151]
		twoHundred = [1 for line in lines if 150<len(line.split())<201]
		twoHundredfifty = [1 for line in lines if 200<len(line.split())<251]
		threeHundred = [1 for line in lines if 250<len(line.split())<301]
		threeHundredfifty = [1 for line in lines if 300<len(line.split())]





		more_40 = [1 for line in lines if len(line.split())>40]
		twenty_25 =[1 for line in lines if 20<len(line.split())<26]
		twentyfive_30 =[1 for line in lines if 25<len(line.split())<31]
		thirty_36 =[1 for line in lines if 30<len(line.split())<36]	
		thirtyfive_40 =[1 for line in lines if 35<len(line.split())<41]	
		lens = [len(zero),len(ten), len(twenty),len(thirty),len(forty),len(fifty),len(sixty),len(sixtyFive),len(seventy), len(seventyFive),len(eighty),len(nighty), len(hundred),len(hundredfifty),len(twoHundred),len(twoHundredfifty),len(threeHundred),len(threeHundredfifty)]
		print('number of all seqs',lens, sum(lens))
		print('number of seq having  20<=length<26  ',len(twenty_25))
		print('number of seq having  25<len(line.split())<31 ',len(twentyfive_30))
		print('number of seq having  30<len(line.split())<36 ',len(thirty_36))
		print('number of seq having  35<len(line.split())<41 ',len(thirtyfive_40))
		print('number of seq having  41<len(line.split())<47 ',len(more_40))

		print("lines smaller &= in len than 3",len(three))
		print("lines smaller &= in len than 2",len(two))
		print("lines smaller &= in len than 1",len(one))
		print(len(lines))


#len_analysis('./GYAFC_Corpus/adding_two_datasets/data_model/data_preprocessed/sentiment.dev.1')
len_analysis('./amazon/amazon_preprocessed/binary_originalSplit_data/sentiment.test.0')
#len_analysis('./paper_news/sentiment.test.1')
#len_analysis('./document-aligned.v2/simple.txt')
#




