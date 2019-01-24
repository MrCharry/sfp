import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Hyperparameters
import Config.config as config

device = torch.device('cuda:0')


class Agent(nn.Module):

    def __init__(self, vocab_size, weight, **kwargs):
        super(Agent, self).__init__(**kwargs)
        # Initialize common parameters
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

    def _loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=config.LR, momentum=config.MOMENTUM)

    def forward(self, x):
        pass

    def _getPreClsIndexes(self, outputs):
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def trainOn(self, X_trainset, y_trainset, batch_size=config.BATCH_SIZE):

        if self.use_gpu:
            self.to(device)
        for epoch in range(config.EPOCHS):
            train_loss = 0
            train_acc = 0
            m = 0
            for X_trains, y_trains in zip(X_trainset, y_trainset):
                # 将数据集且分成K个数据集，进行交叉验证
                # X_trains: 训练集特征集合
                # y_trains: 训练集标签集合
                train_set = torch.utils.data.TensorDataset(X_trains, y_trains)
                train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
                for X_train, y_train in train_iter:
                    m += 1
                    # print(X_train.size(), y_train.size())
                    # if m == 10:
                    #     print('Batch: ', m, '--inputs size: ', X_train.size(), '--labels.size: ', y_train.size())

                    self.optimizer.zero_grad()
                    if self.use_gpu:
                        X_train = X_train.cuda()
                        y_train = y_train.cuda()
                    outputs = self(X_train)
                    loss = self.criterion(outputs, y_train)
                    loss.backward()
                    self.optimizer.step()

                    if self.use_gpu:
                        train_acc += accuracy_score(torch.argmax(outputs.cpu().data, dim=1), y_train.cpu())
                    else:
                        train_acc += accuracy_score(torch.argmax(outputs.data, dim=1), y_train)
                    train_loss += loss.item()

            print('Epoch: ', epoch+1, 'Batch No:  ', m, 'Train loss: ', train_loss/m,
              'train_acc: ', train_acc/m)

    def testOn(self, X_testset, y_testset, batch_size=config.BATCH_SIZE):
        softmax = nn.Softmax()
        with torch.no_grad():
            predicted = []
            n = 0
            for X_tests, y_tests in zip(X_testset, y_testset):
                # 将数据集且分成K个数据集，进行交叉验证
                # X_tests: 训练集特征集合
                # y_tests: 训练集标签集合
                test_set = torch.utils.data.TensorDataset(X_tests, y_tests)
                test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
                for X_test, y_test in test_iter:
                    if self.use_gpu:
                        X_test = X_test.cuda()
                        y_test = y_test.cuda()
                    n += 1
                    outputs = self(X_test)
                    outclass = self._getPreClsIndexes(outputs)
                    if self.use_gpu:
                        outclass = outclass.cpu().numpy()
                    else:
                        outclass = outclass.numpy()
                    predicted.append(outclass)

        return predicted

    def predict(self, x):
        pass
