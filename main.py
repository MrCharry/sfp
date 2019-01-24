from Utility.Utility import Utility
import sys
from ensemble import Ensemble


if __name__ == '__main__':

    util = Utility(sys.argv[1])
    ensemble = Ensemble(util=util)
    ensemble.ensemble_train()
    ensemble.ensemble_test()
