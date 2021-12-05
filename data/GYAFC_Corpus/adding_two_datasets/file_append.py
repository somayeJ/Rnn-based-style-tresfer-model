from os import listdir

file_paths = ['../Entertainment_Music/',  '../Family_Relationships/']
files_mode = 'test'
style = '0'

def file_append(input_files_path,   style, mode):
	'''
	'''
	if style == '0':
		filenames = [f for f in listdir(input_files_path + '/'+ mode) if f[0] =='i'] # 0: informal
	elif style == '1':
		filenames = [f for f in listdir(input_files_path + '/'+ mode) if f[0] == 'f'] # 1: formal
	print( ' style, mode,filenames,input_files_path',style, mode,filenames,input_files_path)
	for filename in filenames:
		fr = open(input_files_path+mode+'/'+ filename)
		data = fr.read()
		fr.close()

		fw = open("./data/sentiment."+mode+"."+style,'a') 
		fw.write(data)
		fw.close()

for path in file_paths:
	file_append(path, style, files_mode)
