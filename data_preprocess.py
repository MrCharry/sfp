import xlrd
import sys
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
import sklearn.metrics as metrics
import matplotlib as plt

def merge_cell(sheet):
	rt = {}
	if sheet.merged_cells:
		#存在合并单元格
		for item in sheet.merged_cells:		
			for row in range(item[0], item[1]):
				for col in range(item[2], item[3]):
					rt.update({(row, col): (item[0], item[2])})
	return rt


def process_merged_cells(filename):	
	book = xlrd.open_workbook(filename)
	sheets = book.sheets()	#获取所有的sheets
	# for index in range(len(sheets)):
	sheet = book.sheet_by_index(2)
	merged_cells = merge_cell(sheet)	#获取所有合并的单元格
	rows = sheet.nrows
	# 如果sheet为空， rows为0
	sheetdata = []
	if rows != 0:
		for row in range(rows):
			rowdata = sheet.row_values(row)	#单行数据
			for index, content in enumerate(rowdata):				
				if merged_cells.get((row, index)):
					#这是合并后的单元格，需要重新取一次数据
					rowdata[index] = sheet.cell_value(*merged_cells.get((row, index)))
			sheetdata.append(rowdata)
	return sheetdata

def retrival_in_sheet(sheetdata, keyword):
	for row in range(len(sheetdata)):
		for col in range(len(sheetdata[row])):
			if sheetdata[row][col] == keyword:
				print(sheetdata[row][0], sheetdata[row][1])
				return sheetdata[row][0], sheetdata[row][1]
	print('没有找到相关信息')


def buildMapOfClass(filename):

	sheetdata = process_merged_cells(filename)
	rows = len(sheetdata)
	# firstclass, secondclass, thirdclass = [],[],[]
	kinds = []
	for row in range(2, rows):
		kind = [sheetdata[row][0], sheetdata[row][1]]
		for i in range(2, len(sheetdata[row])):
			if sheetdata[row][i] != '':
				kinds.append([kind[0], kind[1], sheetdata[row][i]])
	# print(kinds, len(kinds))
	return kinds
	# 	idx = -1
	# 	for i in range(len(firstclass)):
	# 		if firstclass[i] == sheetdata[row][0]:
	# 			idx = i
	# 			break
	# 	if idx>=len(firstclass) or idx==-1:
	# 		firstclass.append(sheetdata[row][0])

	# 	idx = -1
	# 	for i in range(len(secondclass)):
	# 		if secondclass[i] == sheetdata[row][1]:
	# 			idx = i
	# 			break
	# 	if idx>=len(secondclass) or idx==-1:
	# 		secondclass.append(sheetdata[row][1])

	# 	idx = -1
	# 	for i in range(2, len(sheetdata[row])):
	# 		if sheetdata[row][i] != '':
	# 			for j in range(len(thirdclass)):
	# 				if thirdclass[j] == sheetdata[row][i]:
	# 					idx = j
	# 					break
	# 			if idx>=len(thirdclass) or idx==-1:
	# 				thirdclass.append(sheetdata[row][i])
	# return firstclass, secondclass, thirdclass

def shard_data(full_list, shuffle=False, ratio=0.2):
	n_total = len(full_list)
	offset = int(n_total * ratio)
	if n_total==0 or offset<1:
		return [],full_list
	if shuffle:
		random.shuffle(full_list)
	sublist_1 = full_list[:offset]
	sublist_2 = full_list[offset:]
	return sublist_1,sublist_2

def get_data(filename):
	book = xlrd.open_workbook(filename)
	sheets = book.sheets()	#获取所有的sheets
	# for index in range(len(sheets)):
	sheet = book.sheet_by_index(1)
	rows = sheet.nrows
	i,j,k = 0,0,0
	data = []
	kinds = buildMapOfClass(filename)

	for row in range(1, rows):
		rowdata = sheet.row_values(row)
		for i, item in enumerate(kinds):
			if item[0]==rowdata[0] and item[1]==rowdata[1] and item[2]==rowdata[2]:
				data.append([rowdata[3], i])

	# for row in range(1, rows):
	# 	rowdata = sheet.row_values(row)
	# 	target = []
	# 	for index, content in enumerate(rowdata):
	# 		if index == 0:
	# 			for m in range(len1):
	# 				if firstclass[m] == content:
	# 					i = m
	# 		if index == 1:
	# 			for m in range(len2):
	# 				if secondclass[m] == content:
	# 					j = m
	# 		if index == 2:
	# 			for m in range(len3):
	# 				if thirdclass[m] == content:
	# 					k = m		
	# 	target[i] = 1
	# 	target[len1+j] = 1
	# 	target[len1+len2+k] = 1
	# 	data.append([rowdata[3], target])
	# 80%为训练数据，20%为测试数据，采用随机划分
	traindata, testdata = shard_data(data, shuffle=True, ratio=0.8)
	return traindata, testdata, kinds

def tokenizer(text):
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


def tokenize_data(traindata, testdata):
	train_tokenized = []
	test_tokenized = []
	for review, tag in traindata:
		train_tokenized.append(tokenizer(review))
	for review, tag in testdata:
		test_tokenized.append(tokenizer(review))

	#生成训练集向量词库
	vocab = []
	for review in train_tokenized:
		for phrase in review.split(' '):
			if phrase not in vocab:
				vocab.append(phrase)
	vocab_size = len(vocab)
	return train_tokenized, test_tokenized, vocab, vocab_size

def encode_samples(tokenized_samples, vocab):

	word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'

	features = []
	for sample in tokenized_samples:
		feature = []		
		for token in sample.split(' '):
			if token in word_to_idx:
				feature.append(word_to_idx[token])
			else:
				feature.append(0)
		features.append(feature)
	return features


def pad_samples(features, maxlen=500, PAD=0):
	#规整样例，所有评论只取前面500个词，不足补0
	padded_features = []
	for feature in features:
		if len(feature) >= maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature = feature
			while len(padded_feature) < maxlen:
				padded_feature.append(PAD)
		padded_features.append(padded_feature)
	return padded_features


def data_for_train(traindata, testdata):
	train_tokenized, test_tokenized, vocab, vocab_size = tokenize_data(traindata, testdata)
	train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
	train_labels = torch.tensor([tag for _, tag in traindata])
	test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
	test_labels = torch.tensor([tag for _, tag in testdata])

	return train_features, train_labels, test_features, test_labels, vocab, vocab_size


def get_weight(wvmodel, vocab, vocab_size, embed_size):

	word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'
	weight = torch.zeros(vocab_size+1,  embed_size)

	for i in range(len(wvmodel.index2word)):
		try:
			index = word_to_idx[wvmodel.index2word[i]]			
		except:
			continue
		weight[index,:] = torch.from_numpy(wvmodel.get_vector(
			idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
	return weight


def tag2Desc(output, labels, kinds):
	
	score = 0
	desc = []
	for (plabel, label) in zip(output, labels):
		plabel = plabel.tolist()
		label = label.tolist()
		pindex = plabel.index(max(plabel))
		index = label
		if pindex == index:
			score += 1

		desc.append([kinds[pindex], kinds[index]])

	return score, desc


class SentimentNet(nn.Module):
	"""docstring for SentimentNet"""
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 bidirectional, weight, labels, use_gpu, **kwargs):
		super(SentimentNet, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.num_layers = num_layers
		self.use_gpu = use_gpu
		self.bidirectional = bidirectional
		self.embedding = nn.Embedding.from_pretrained(weight)
		self.embedding.weight.requires_grad = False
		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
								num_layers=num_layers, bidirectional=self.bidirectional,
								dropout=0)
		if self.bidirectional:
			self.decoder = nn.Linear(num_hiddens*4, labels)
		else:
			self.decoder = nn.Linear(num_hiddens*2, labels)

	def forward(self, inputs):
		embeddings = self.embedding(inputs)
		states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
		encoding = torch.cat([states[0], states[-1]], dim=1)
		outputs = self.decoder(encoding)
		return outputs


class textCNN(nn.Module):
	def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
		super(textCNN, self).__init__(**kwargs)
		self.labels = labels
		self.embedding = nn.Embedding.from_pretrained(weight)
		self.embedding.weight.requires_grad = False
		self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
		self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
		self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
		self.pool1 = nn.MaxPool2d((seq_len-3+1, 1))
		self.pool2 = nn.MaxPool2d((seq_len-4+1, 1))
		self.pool3 = nn.MaxPool2d((seq_len-5+1, 1))
		self.linear = nn.Linear(3, labels)

	def forward(self, inputs):
		inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
		x1 = F.relu(self.conv1(inputs))
		x2 = F.relu(self.conv2(inputs))
		x3 = F.relu(self.conv3(inputs))

		x1 = self.pool1(x1)
		x2 = self.pool2(x2)
		x3 = self.pool3(x3)

		x = torch.cat((x1, x2, x3), -1)
		x = x.view(inputs.shape[0], 1, -1)

		x = self.linear(x)
		x = x.view(-1, self.labels)

		return x


#Hyperparameters
num_epochs = 2
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
batch_size = 32
labels = 133
lr = 0.2
use_gpu = False


if __name__ == '__main__':
	filename = sys.argv[1]
	traindata, testdata, kinds = get_data(filename)
	train_features, train_labels, test_features, test_labels, vocab, vocab_size = data_for_train(traindata, testdata)
	wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./phrase.vector', binary=False, encoding='utf-8')
	weight = get_weight(wvmodel, vocab, vocab_size, embed_size)

	net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
						num_hiddens=num_hiddens, num_layers=num_layers,
						bidirectional=bidirectional, weight=weight,
						labels=labels, use_gpu=use_gpu)
	# net = textCNN(vocab_size=vocab_size+1, embed_size=embed_size,
	# 	seq_len=500, labels=labels, weight=weight)

	# loss_function = nn.BCELoss()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr)
	# optimizer = optim.Adam(net.parameters(), lr=lr)
	train_set = torch.utils.data.TensorDataset(train_features, train_labels)
	test_set = torch.utils.data.TensorDataset(test_features, test_labels)

	train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

	softmax = nn.Softmax()
	for epoch in range(num_epochs):
		start = time.time()
		n, m = 0, 0
		# scores = 0
		# descs = []
		train_losses, test_losses = 0, 0
		train_acc, test_acc = 0, 0
		for feature, label in train_iter:
			n += 1			
			optimizer.zero_grad()
			feature = Variable(feature)
			label = Variable(label)
			score = net(feature)
			loss = loss_function(score, label)
			loss.backward()
			optimizer.step()
			train_acc += metrics.accuracy_score(score.argmax(dim=1), label)
			train_losses += loss
			# if n%10 == 0:
			# 	print('Train batch %d'%n, 'loss: {}'.format(loss))

		with torch.no_grad():
			for test_feature, test_label in test_iter:
				m += 1				
				test_score = net(test_feature)
				test_loss = loss_function(test_score, test_label)
				test_acc += metrics.accuracy_score(test_score.argmax(dim=1), test_label)
				test_losses += test_loss
				# score, desc = tag2Desc(softmax(output), test_label, kinds)
				# scores += score
				# descs.append(desc)
				# if m%10 == 0:
				# 	print('Test batch %d'%m, 'loss: {}'.format(test_loss))
		end = time.time()
		runtime = end - start
		print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
          (epoch, train_losses / n, train_acc / n, test_losses / m, test_acc / m, runtime))
		# accuracy = scores / (m*batch_size)
		# print('Accuracy: %.2f' % accuracy, 'Epoch %d runtime: %.2f' % (epoch, runtime))