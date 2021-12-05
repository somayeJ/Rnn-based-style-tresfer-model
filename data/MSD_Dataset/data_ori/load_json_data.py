import json
import pandas as pd

with open('train.txt','r')  as f, open("sentiment.train_ori.0","w") as ts0, open("sentiment.train_ori.1","w") as ts1:

	data = [json.loads(line) for line in f]
	print('done')
  
	# Iterating through the json_list 
	
	expert0=[]
	layman1=[]
	for d in data: 
		if d['label'] == 0:
			expert0.append(d['text']) 
			ts0.write(d['text'])
			ts0.write('\n')
		elif d['label'] == 1:
			layman1.append(d['text'])
			ts1.write(d['text'])
			ts1.write('\n')
		else:
			print("a label different from 0 & 1")
	  
