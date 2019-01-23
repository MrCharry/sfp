import Model.AgentSet as Agents
import Utility.utility as util
import sys


if __name__ == '__main__':

    util = util.Utility(sys.argv[1])
    # agent = Agents.LSTM(vocab_size=util.vocab_size, weight=util.weight)
    agent = Agents.CNN(vocab_size=util.vocab_size, weight=util.weight)
    agent.trainOn(X_trainset=util.X_trainset, y_trainset=util.y_trainset)
    agent.testOn(X_testset=util.X_testset, y_testset=util.y_testset)
