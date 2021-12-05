# this code merges file1 & 2 writes in file3, as follows:
file_read_1 = 'sentiment.train_ori.0'
file_read_2 = 'sentiment.test.0'
file_write = 'data_ori.0'

def merge_file(file_read_1,file_read_2,file_write):

	with open(file_read_1,'r')  as r1, open(file_read_2,"r") as r2, open(file_write,"w") as w:
		data = r1.readlines()
		data. extend(r2.readlines())
		print(len(data))
		for line in data: 
			w.write(line.strip())
			w.write('\n')

		  
merge_file(file_read_1,file_read_2,file_write)
