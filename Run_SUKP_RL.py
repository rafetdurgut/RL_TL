# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:18:05 2021

@author: ASUS
"""

from itertools import product
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from AOS import *
from Utilities import Log
import time

import sys
pNo = int(sys.argv[1])
problem = SetUnionKnapsack('Data/SUKP', pNo)
runtime = 2
operator_pool = [disABC(0.9, 0.1), ibinABC(0.3, 0.1),  binABC()]

operator_selectors = [
       ClusterRL(len(operator_pool), 'extreme', 25,0.5, 0.1,learning_mode=0),
       ClusterRL(len(operator_pool), 'extreme', 25,0.5, 0.1,learning_mode=1),
       ClusterRL(len(operator_pool), 'extreme', 25,0.5, 0.1,learning_mode=2)
       # ProbabilityMatching(len(operator_pool),'average',W=10,learning_mode=1),
       # ProbabilityMatching(len(operator_pool),'average',W=10,learning_mode=2)
       
       ]
for operator_selector in operator_selectors: 
    for run in range(runtime):
        start_time = time.time()
        elapsed_time = []

        abc = BinaryABC(problem, operator_pool, operator_selector, pop_size=20, maxFE=1*max(problem.m, problem.n),limit=100)
        for operator in operator_pool:
            operator.set_algorithm(abc)
        operator_selector.set_algorithm(abc, run)

        abc.run()
        # your code
        elapsed_time.append( [time.time() - start_time])
        time_logs = Log(elapsed_time, 'results', 'time', abc.operator_selector.__conf__(),problem.dosyaAdi)
        convergence_logs = Log(abc.convergence, 'results', 'cg', abc.operator_selector.__conf__(),problem.dosyaAdi)
        credit_logs = Log(abc.operator_selector.credits, 'results', 'credits', abc.operator_selector.__conf__(),problem.dosyaAdi)
        reward_logs = Log(abc.operator_selector.rewards, 'results', 'rewards', abc.operator_selector.__conf__(),problem.dosyaAdi)
        usage_logs = Log(abc.operator_selector.usage_counter, 'results', 'usage', abc.operator_selector.__conf__(),problem.dosyaAdi)
        success_logs = Log(abc.operator_selector.success_counter, 'results', 'success', abc.operator_selector.__conf__(),problem.dosyaAdi)