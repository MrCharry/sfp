import Config.config as config
import numpy as np
import Agent.AgentSet as Agents
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tkinter
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score, auc


class Ensemble(object):

    def __init__(self, util, **kwargs):
        super(Ensemble, self).__init__(**kwargs)
        self.util = util

    def ensemble_train(self):

        util = self.util
        # agents = util.model_loader(util.modelpath)
        # if len(agents) == 0:
        #     # 模型加载失败，重新训练模型
        #     print('Begin retraining...')
        #     # 分别在数据集上训练出CNN和LSTM模型
        #     agents = []
        for i in range(config.AGENT_NUM):
            # 每次采取不同的初始化矩阵进行训练
            util.init_weight()
            print('The ', i+1, ' round: ')
            agentCNN = Agents.CNN(vocab_size=util.vocab_size, weight=util.weight)
            agentLSTM = Agents.LSTM(vocab_size=util.vocab_size, weight=util.weight)
            agentCNN.trainOn(X_trainset=util.X_trainset, y_trainset=util.y_trainset)
            agentLSTM.trainOn(X_trainset=util.X_trainset, y_trainset=util.y_trainset)
            agents.append(agentCNN)
            agents.append(agentLSTM)
            # 将训练好的模型保存在agents里
            torch.save(agentCNN, './Model/model'+str(i*2+1)+'.pth')
            torch.save(agentLSTM, './Model/model'+str(i*2+2)+'.pth')
        # self.agents = agents
        print('Training finished...')
        # else:
        #     self.agents = agents

    def ensemble_test(self):

        # 开始测试过程
        predictedset = []
        # batch_size = config.BATCH_SIZE
        util = self.util
        agents = util.model_loader(util.modelpath)

        for agent in agents:
            predicted = agent.testOn(X_testset=util.X_testset, y_testset=util.y_testset)
            predictedset.append(predicted)

        # 拼接目标类标
        targets = []
        for y_tests in util.y_testset:
            targets += y_tests.numpy().tolist()

        correct = 0
        i = -config.BATCH_SIZE
        class_correct = list(0. for i in range(config.CLASS_NUM))
        class_total = list(0. for i in range(config.CLASS_NUM))
        ensemble_predicteds = []
        # confusion_matrix = np.zeros((config.CLASS_NUM, config.CLASS_NUM))
        # predictedset 包含10个predicted集合，分别为5个CNN和5个LSTM在测试集上的预测结果
        # 遍历10个predicted集合，对预测标签进行打分，取得分最高的类标作为集成模型的输出
        for p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 in zip(predictedset[0], predictedset[1], predictedset[2],
                                                          predictedset[3],
                                                          predictedset[4], predictedset[5], predictedset[6],
                                                          predictedset[7],
                                                          predictedset[8], predictedset[9]):
            # p1~p10为长度为batch_size的对应10个模型的预测输出, size分别为1*batch_size
            # 将p1~p10拼接为10*batch_size的矩阵，目的是求每一列出现次数最多的数作为预测输出
            # 为了方便，将矩阵转置为batch_size*10的矩阵，每行出现次数最多的即为输出的预测类标，每次循环输出batch_size个预测值
            # p = torch.cat([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9], dim=0).t() # p为batch_size*10的矩阵
            p = np.c_[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]  # 将10个batch_sizes数组按列拼接组成32*10的矩阵
            if p.shape[0] < config.BATCH_SIZE:
                i += p.shape[0]
            else:
                i += config.BATCH_SIZE
            for j, item in enumerate(p):
                # 遍历所有训练好的模型，对每个模型的训练结果进行投票，选出最佳的结果保存在ensemble_predicted
                ensemble_predicted = util.max_appearance_in_list(item.tolist())
                ensemble_predicteds.append(ensemble_predicted)
                if ensemble_predicted == targets[i+j]:
                    correct += 1
                    class_correct[ensemble_predicted] += 1
                class_total[ensemble_predicted] += 1
        # print('Accuracy of the Ensemble Network: ', correct / i, 'Total samples: ', i)

        # 输出每个类目判断的准确度
        with open('./Out/out.txt', 'a') as f:
            # 所有样本中被预测正确的样本的比率（包含了所有class的总体准确率）
            f.write('Accuracy of the all samples with Ensemble Network: %.2f\n'
                    %(accuracy_score(targets, ensemble_predicteds)))
            f.write('Accuracy\tPrecision\tRecall\tF1\t\n')
            for k in range(config.CLASS_NUM):
                if class_total[k] == 0:
                    class_correct_rate = 0
                else:
                    class_correct_rate = class_correct[k] / class_total[k]
                print('Accuracy of ', util.CLASSES[k], ': ', class_correct_rate)
                f.write('%s %.2f\n' % (util.CLASSES[k], class_correct_rate))

        # 画出预测结果的混淆矩阵
        cmat = confusion_matrix(ensemble_predicteds, targets)
        util.draw_confusion_matrix(cmat, './Out/confusion_matrix.jpeg')

    def ensemble_prediction(self):
        pass
