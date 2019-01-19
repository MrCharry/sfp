# coding: utf-8
# -- delete stopword in content --
import sys
import jieba

def clearStopwords(text):
	wordlist = []
	seg_list = jieba.cut(text, cut_all=False)
	liststr = '/'.join(seg_list)
	f_stop = open('stopwords.txt')
	try:
		f_stop_text = f_stop.read()
		f_stop_text = str(f_stop_text)
	finally:
		f_stop.close()
	f_stop_seg_list = f_stop_text.split('\n')
	for word in liststr.split('/'):
		if not(word.strip() in f_stop_seg_list) and len(word.strip())>1:
			wordlist.append(word)
	return ' '.join(wordlist)

if __name__ == '__main__':
	inputfile, outputfile = sys.argv[1:3]
	text = open(inputfile).read()
	cut_words = clearStopwords(text)
	with open(outputfile, 'w', encoding='utf-8') as f:
		f.write(cut_words)