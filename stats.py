# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:19:30 2021

@author: Win7
"""

import pandas as pd
import numpy as np
data = pd.read_csv('Analysis/set3.csv',header=None,delimiter=';')

from scipy.stats import rankdata
ranks_mean = np.zeros((10,7))
ranks_best = np.zeros((10,7))
aa = []
bb = []
for i in range(10):
    sira_verib = data.iloc[i*3,:].values.tolist()
    sira_veri = data.iloc[i*3+1,:].values.tolist()
    aa.append(sira_veri)
    bb.append(sira_verib)
    sira_veri = [-1* x for x in sira_veri]
    sira_verib = [-1* x for x in sira_verib]
    
    ranks_mean[i] = rankdata(sira_veri,method='min')
    ranks_best[i] = rankdata(sira_verib,method='min')
    
mean_rank = np.mean(ranks_mean,axis=0)
best_rank = np.mean(ranks_best,axis=0)


numpy_array = np.array(aa)
transpose = numpy_array.T
aaa = transpose.tolist()


numpy_array = np.array(bb)
transpose = numpy_array.T
bbb = transpose.tolist()


ps = []
w, p = friedmanchisquare(aaa[0],aaa[1],aaa[2],aaa[3],aaa[4],aaa[5],aaa[6])
ps.append(p)

w, p = friedmanchisquare(bbb[0],bbb[1],bbb[2],bbb[3],bbb[4],bbb[5],bbb[6])
ps.append(p)

