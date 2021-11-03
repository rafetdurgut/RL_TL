# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:53:49 2021

@author: Win7
"""

from itertools import product
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from Utilities import Log
import sys
c=dict()
c["Method"] = "extreme"
c["W"] = 25
c["Pmin"] = 0.1
c["Alpha"] = 0.1
c["Gama"] = 0.3
pNo = int(sys.argv[1])
problem = SetUnionKnapsack('Data/SUKP', pNo)
runtime = 100
operator_pool = [disABC(0.9, 0.1), ibinABC(0.3, 0.1),  binABC()]
for run in range(runtime):
    operator_selectors = [
        ClusterRL(len(operator_pool), reward_type=c["Method"], W=c["W"], alpha=c["Alpha"],
                        Pmin=c["Pmin"], gama=c["Gama"])]
    for operator_selector in operator_selectors:

        abc = BinaryABC(problem, operator_pool, operator_selector, pop_size=20, maxFE=40*max(problem.m, problem.n),
                        limit=100)
        for operator in operator_pool:
            operator.set_algorithm(abc)
        operator_selector.set_algorithm(abc)
        abc.run()
        convergence_logs = Log(abc.convergence, 'results', 'cg', abc.operator_selector.__conf__(),pNo)
        credit_logs = Log(abc.operator_selector.credits, 'results', 'credits', abc.operator_selector.__conf__(),pNo)
        reward_logs = Log(abc.operator_selector.rewards, 'results', 'rewards', abc.operator_selector.__conf__(),pNo)
        usage_logs = Log(abc.operator_selector.usage_counter, 'results', 'usage', abc.operator_selector.__conf__(),pNo)
        success_logs = Log(abc.operator_selector.success_counter, 'results', 'success', abc.operator_selector.__conf__(),pNo)