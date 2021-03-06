import xlrd
import jieba
import torch
from sklearn.model_selection import KFold
import numpy as np
import Config.config as config
import gensim

class Utility(object):

	def __init__(self, filename):
		self.filename = filename
		self.wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./phrase.vector', binary=False, encoding='utf-8')
		self.data_for_train()

	def merge_cell(self, sheet):
		rt = {}
		if sheet.merged_cells:
			# 存在合并单元格12`
			for item in sheet.merged_cells:
				for row in range(item[0], item[1]):
					for col in range(item[2], item[3]):
						rt.update({(row, col): (item[0], item[2])})
		return rt

	def process_merged_cells(self):

		book = xlrd.open_workbook(self.filename)
		sheets = book.sheets()  # 获取所有的sheets
		# for index in range(len(sheets)):
		sheet = book.sheet_by_index(2)
		merged_cells = self.merge_cell(sheet)  # 获取所有合并的单元格
		rows = sheet.nrows
		# 如果sheet为空， rows为0
		sheetdata = []
		if rows != 0:
			for row in range(rows):
				rowdata = sheet.row_values(row)  # 单行数据
				for index, content in enumerate(rowdata):
					if merged_cells.get((row, index)):
						# 这是合并后的单元格，需要重新取一次数据
						rowdata[index] = sheet.cell_value(*merged_cells.get((row, index)))
				sheetdata.append(rowdata)
		return sheetdata

	@staticmethod
	def retrival_in_sheet(self, keyword):

		sheetdata = self.process_merged_cells()
		for row in range(len(sheetdata)):
			for col in range(len(sheetdata[row])):
				if sheetdata[row][col] == keyword:
					print(sheetdata[row][0], sheetdata[row][1])
					return sheetdata[row][0], sheetdata[row][1]
		print('没有找到相关信息')

	def build_map_of_class(self):

		sheetdata = self.process_merged_cells()
		rows = len(sheetdata)
		kinds = []
		for row in range(2, rows):
			kind = [sheetdata[row][0], sheetdata[row][1]]
			for i in range(2, len(sheetdata[row])):
				if sheetdata[row][i] != '':
					kinds.append([kind[0], kind[1], sheetdata[row][i]])
		# print(kinds, len(kinds))
		self.CLASSES = kinds
		return kinds

	def shard_data(self, K=config.K):

		features = np.array(self.features)
		labels = np.array(self.labels)
		X_trainset, y_trainset, X_testset, y_testset = [],[],[],[]
		kf = KFold(n_splits=K, shuffle=True, random_state=0)
		for train_index, test_index in kf.split(features, labels):
			X_trainset.append(torch.tensor(features[train_index]))
			y_trainset.append(torch.tensor(labels[train_index]))
			X_testset.append(torch.tensor(features[test_index]))
			y_testset.append(torch.tensor(labels[test_index]))
		self.X_trainset = X_trainset
		self.y_trainset = y_trainset
		self.X_testset = X_testset
		self.y_testset = y_testset

	def get_natural_data(self):

		book = xlrd.open_workbook(self.filename)
		sheets = book.sheets()  # 获取所有的sheets
		# for index in range(len(sheets)):
		sheet = book.sheet_by_index(1)
		rows = sheet.nrows
		i, j, k = 0, 0, 0
		data = []
		kinds = self.build_map_of_class()

		for row in range(1, rows):
			rowdata = sheet.row_values(row)
			for i, item in enumerate(kinds):
				if item[0] == rowdata[0] and item[1] == rowdata[1] and item[2] == rowdata[2]:
					data.append([rowdata[3], i])

		return data, kinds

	def tokenizer(self, text):
		wordlist = []
		seg_list = jieba.cut(text, cut_all=False)
		liststr = '/'.join(seg_list)
		f_stop = open('./stopwords.txt')
		try:
			f_stop_text = f_stop.read()
			f_stop_text = str(f_stop_text)
		finally:
			f_stop.close()
		f_stop_seg_list = f_stop_text.split('\n')
		for word in liststr.split('/'):
			if not (word.strip() in f_stop_seg_list) and len(word.strip()) > 1:
				wordlist.append(word)
		return ' '.join(wordlist)

	def tokenize_data(self):

		data_tokenized = []
		data, _ = self.get_natural_data()
		for review, tag in data:
			data_tokenized.append([self.tokenizer(review), tag])

		# 生成数据集向量词库
		vocab = []
		for review, _ in data_tokenized:
			for phrase in review.split(' '):
				if phrase not in vocab:
					vocab.append(phrase)
		vocab_size = len(vocab)
		self.data_tokenized = data_tokenized
		self.vocab = vocab
		self.vocab_size = vocab_size

	def encode_samples(self):

		vocab = self.vocab
		word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
		word_to_idx['<unk>'] = 0
		idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
		idx_to_word[0] = '<unk>'

		features = []
		for sample, _ in self.data_tokenized:
			feature = []
			for token in sample.split(' '):
				if token in word_to_idx:
					feature.append(word_to_idx[token])
				else:
					feature.append(0)
			features.append(feature)
		self.features = features

	def pad_samples(self, maxlen=config.MAX_LEN, PAD=config.PAD):
		# 规整样例，所有评论只取前面MAX_LEN个词，不足补PAD
		padded_features = []
		vocab_size = self.vocab_size
		for feature in self.features:
			if len(feature) >= maxlen:
				padded_feature = feature[:maxlen]
			else:
				padded_feature = feature
				while len(padded_feature) < maxlen:
					padded_feature.append(PAD)
			padded_features.append(padded_feature)
		self.features = padded_features

	def data_for_train(self):
		self.tokenize_data()
		self.encode_samples()
		self.pad_samples()
		self.labels = [tag for _, tag in self.data_tokenized]
		self.shard_data()
		self.get_weight()
		# return train_features, train_labels, test_features, test_labels, vocab, vocab_size

	def get_weight(self):

		vocab, vocab_size, embed_size = self.vocab, self.vocab_size, config.EMBED_SIZE
		wvmodel = self.wvmodel
		word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
		word_to_idx['<unk>'] = 0
		idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
		idx_to_word[0] = '<unk>'
		weight = torch.zeros(vocab_size + 1, embed_size)

		for i in range(len(wvmodel.index2word)):
			try:
				index = word_to_idx[wvmodel.index2word[i]]
			except:
				continue
			weight[index, :] = torch.from_numpy(wvmodel.get_vector(
				idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
		self.weight = weight
	#
	# def tag2Desc(output, labels, kinds):
	#
	# 	score = 0
	# 	desc = []
	# 	for (plabel, label) in zip(output, labels):
	# 		plabel = plabel.tolist()
	# 		label = label.tolist()
	# 		pindex = plabel.index(max(plabel))
	# 		index = label
	# 		if pindex == index:
	# 			score += 1
	#
	# 		desc.append([kinds[pindex], kinds[index]])
	#
	# 	return score, desc


#
# class SentimentNet(nn.Module):
# 	"""docstring for SentimentNet"""
# 	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
# 				 bidirectional, weight, labels, use_gpu, **kwargs):
# 		super(SentimentNet, self).__init__(**kwargs)
# 		self.num_hiddens = num_hiddens
# 		self.num_layers = num_layers
# 		self.use_gpu = use_gpu
# 		self.bidirectional = bidirectional
# 		self.embedding = nn.Embedding.from_pretrained(weight)
# 		self.embedding.weight.requires_grad = False
# 		self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
# 								num_layers=num_layers, bidirectional=self.bidirectional,
# 								dropout=0)
# 		if self.bidirectional:
# 			self.decoder = nn.Linear(num_hiddens*4, labels)
# 		else:
# 			self.decoder = nn.Linear(num_hiddens*2, labels)
#
# 	def forward(self, inputs):
# 		embeddings = self.embedding(inputs)
# 		states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
# 		encoding = torch.cat([states[0], states[-1]], dim=1)
# 		outputs = self.decoder(encoding)
# 		return outputs
#
#
# class textCNN(nn.Module):
# 	def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
# 		super(textCNN, self).__init__(**kwargs)
# 		self.labels = labels
# 		self.embedding = nn.Embedding.from_pretrained(weight)
# 		self.embedding.weight.requires_grad = False
# 		self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
# 		self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
# 		self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
# 		self.pool1 = nn.MaxPool2d((seq_len-3+1, 1))
# 		self.pool2 = nn.MaxPool2d((seq_len-4+1, 1))
# 		self.pool3 = nn.MaxPool2d((seq_len-5+1, 1))
# 		self.linear = nn.Linear(3, labels)
#
# 	def forward(self, inputs):
# 		inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
# 		x1 = F.relu(self.conv1(inputs))
# 		x2 = F.relu(self.conv2(inputs))
# 		x3 = F.relu(self.conv3(inputs))
#
# 		x1 = self.pool1(x1)
# 		x2 = self.pool2(x2)
# 		x3 = self.pool3(x3)
#
# 		x = torch.cat((x1, x2, x3), -1)
# 		x = x.view(inputs.shape[0], 1, -1)
#
# 		x = self.linear(x)
# 		x = x.view(-1, self.labels)
#
# 		return x


#Hyperparameters
# num_epochs = 2
# embed_size = 100
# num_hiddens = 100
# num_layers = 2
# bidirectional = True
# batch_size = 32
# labels = 133
# lr = 0.2
# use_gpu = False


# if __name__ == '__main__':
# 	filename = sys.argv[1]
# 	utility = Utility(filename)
# 	for i, item in enumerate(utility.CLASSES):
# 		print('第', i, '类：', item)
	# traindata, testdata, kinds = get_data(filename)
	# train_features, train_labels, test_features, test_labels, vocab, vocab_size = data_for_train(traindata, testdata)
	# wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./phrase.vector', binary=False, encoding='utf-8')
	# weight = get_weight(wvmodel, vocab, vocab_size, embed_size)
	#
	# net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
	# 					num_hiddens=num_hiddens, num_layers=num_layers,
	# 					bidirectional=bidirectional, weight=weight,
	# 					labels=labels, use_gpu=use_gpu)
	# net = textCNN(vocab_size=vocab_size+1, embed_size=embed_size,
	# 	seq_len=500, labels=labels, weight=weight)
	#
	# loss_function = nn.BCELoss()
	# loss_function = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=lr)
	# optimizer = optim.Adam(net.parameters(), lr=lr)
	# train_set = torch.utils.data.TensorDataset(train_features, train_labels)
	# test_set = torch.utils.data.TensorDataset(test_features, test_labels)
	#
	# train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
	# test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
	#
	# softmax = nn.Softmax()
	# for epoch in range(num_epochs):
	# 	start = time.time()
	# 	n, m = 0, 0
	# 	train_losses, test_losses = 0, 0
	# 	train_acc, test_acc = 0, 0
	# 	for feature, label in train_iter:
	# 		n += 1
	# 		optimizer.zero_grad()
	# 		feature = Variable(feature)
	# 		label = Variable(label)
	# 		score = net(feature)
	# 		loss = loss_function(score, label)
	# 		loss.backward()
	# 		optimizer.step()
	# 		train_acc += metrics.accuracy_score(score.argmax(dim=1), label)
	# 		train_losses += loss
	#
	# 	with torch.no_grad():
	# 		for test_feature, test_label in test_iter:
	# 			m += 1
	# 			test_score = net(test_feature)
	# 			test_loss = loss_function(test_score, test_label)
	# 			test_acc += metrics.accuracy_score(test_score.argmax(dim=1), test_label)
	# 			test_losses += test_loss
	#
	# 	end = time.time()
	# 	runtime = end - start
	# 	print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
	# 		  (epoch, train_losses / n, train_acc / n, test_losses / m, test_acc / m, runtime))