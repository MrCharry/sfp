from Model.Agent import Agent
import torch.nn as nn
import torch.nn.functional as F
import torch
import Config.config as config
import torch.optim as optim


class CNN(Agent):

    def __init__(self, vocab_size, weight, seq_len=500, lr=config.LR, momentum=config.MOMENTUM, use_gpu=False, **kwargs):
        super(CNN, self).__init__(vocab_size, weight)
        self.seq_len = seq_len
        self.lr = lr
        self.momentum = momentum
        self.use_gpu = use_gpu
        self.conv1 = nn.Conv2d(1, 1, (3, config.EMBED_SIZE))
        self.conv2 = nn.Conv2d(1, 1, (4, config.EMBED_SIZE))
        self.conv3 = nn.Conv2d(1, 1, (5, config.EMBED_SIZE))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, config.CLASS_NUM)
        self._loss_function()

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], 1, x.shape[1], -1)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(x.shape[0], 1, -1)

        x = self.linear(x)
        x = x.view(-1, config.CLASS_NUM)

        return x

    def _loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)


class LSTM(Agent):

    def __init__(self, vocab_size, weight, num_hiddens=100, num_layers=2, bidirectional=True, use_gpu=True, lr=config.LR,
                 momentum=config.MOMENTUM, dropout=0.5, **kwargs):

        super(LSTM, self).__init__(vocab_size, weight)
        self.lr = lr
        self.momentum = momentum
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.num_hiddens = num_hiddens
        self.encoder = nn.LSTM(input_size=config.EMBED_SIZE, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=dropout)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, config.CLASS_NUM)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, config.CLASS_NUM)
        self._loss_function()

    def _loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

    def forward(self, x):
        embeddings = self.embedding(x)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs

