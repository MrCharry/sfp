import Utility.Utility as util
import sys
from ensemble import Ensemble


if __name__ == '__main__':

    util = util.Utility(sys.argv[1])
    ensemble = Ensemble(util=util)
    ensemble.ensemble_train()
    ensemble.ensemble_test()
