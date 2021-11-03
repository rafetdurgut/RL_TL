from itertools import product
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from Utilities import Log
import sys
c = {"Method": "average", "W": 5, "Pmin": 0.1, "Alpha": 0.9}
# c=dict()
# c["Method"] = sys.argv[1]
# c["W"] = int(sys.argv[2])
# c["Pmin"] = float(sys.argv[3])
# c["Alpha"] = float(sys.argv[4])

problem = ZeroOneKnapsack('Data/01KP', 6)
runtime = 2
operator_pool = [disABC(0.9, 0.1), ibinABC(0.3, 0.1),  binABC(), nABC(), flipABC()]
for run in range(runtime):
    operator_selectors = [
        ProbabilityMatching(len(operator_pool), reward_type=c["Method"], W=c["W"], alpha=c["Alpha"],
                        Pmin=c["Pmin"]),
        AdaptivePursuit(len(operator_pool), reward_type=c["Method"], W=c["W"], alpha=c["Alpha"],
                       Pmin=c["Pmin"]),
        UpperConfidenceBound(len(operator_pool), reward_type=c["Method"], W=c["W"], alpha=c["Alpha"],
                        Pmin=c["Pmin"]),
        ClusterRL(len(operator_pool), reward_type=c["Method"], W=c["W"], beta=c["Alpha"],
                        Pmin=c["Pmin"])]
    for operator_selector in operator_selectors:

        abc = BinaryABC(problem, operator_pool, operator_selector, pop_size=20, maxFE=5000,
                        limit=500)
        for operator in operator_pool:
            operator.set_algorithm(abc)
        operator_selector.set_algorithm(abc)
        abc.run()
        convergence_logs = Log(abc.convergence, 'results', 'cg', abc.operator_selector.__conf__())
        if abc.operator_selector.type == 'iteration':
            credit_logs = Log(abc.operator_selector.credits, 'results', 'credits', abc.operator_selector.__conf__())
            reward_logs = Log(abc.operator_selector.rewards, 'results', 'rewards', abc.operator_selector.__conf__())

        else:
            credit_logs = Log(abc.operator_selector.iter_credits, 'results', 'credits', abc.operator_selector.__conf__())
            reward_logs = Log(abc.operator_selector.iter_rewards, 'results', 'rewards', abc.operator_selector.__conf__())

        usage_logs = Log(abc.operator_selector.usage_counter, 'results', 'usage', abc.operator_selector.__conf__())
        success_logs = Log(abc.operator_selector.success_counter, 'results', 'success', abc.operator_selector.__conf__())