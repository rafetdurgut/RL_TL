from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from Utilities import Log
if __name__ == '__main__':
    problem = ZeroOneKnapsack('Data/01KP/large',0)
    operator_pool = [disABC(0.9, 0.1), ibinABC(0.3, 0.1), flipABC(), twoOptABC(), GBABC()]
    operator_selector = ProbabilityMatching(len(operator_pool), 'average',alpha=0.9, W=25, Pmin=0.1)
    abc = BinaryABC(problem, operator_pool, operator_selector, pop_size=20, maxFE=10000, limit=100)
    for operator in operator_pool:
        operator.set_algorithm(abc)
    operator_selector.set_algorithm(abc)
    abc.run()
    credit_logs = Log(abc.operator_selector.credits, 'results', 'credits', abc.operator_selector.__conf__())
    reward_logs = Log(abc.operator_selector.rewards, 'results', 'rewards', abc.operator_selector.__conf__())
    usage_logs = Log(abc.operator_selector.usage_counter, 'results', 'usage', abc.operator_selector.__conf__())
    success_logs = Log(abc.operator_selector.success_counter, 'results', 'success', abc.operator_selector.__conf__())
    print(abc.convergence)
