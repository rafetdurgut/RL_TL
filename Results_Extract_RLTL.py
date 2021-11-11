# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:35:37 2021

@author: Win7
"""

import csv
from itertools import product
from Problem import SetUnionKnapsack
def get_best_data(fileName, operator_size):
    datas = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        previous_iter = -1
        previous_val = 0
        for row in csv_reader:
            if len(row) > 0:
                iteration, val = row
                
                
                if iteration <= previous_iter:
                    datas.append((previous_val))
                    previous_val = val
                    previous_iter = 0
                else:
                    previous_val = val
                    previous_iter = iteration
        datas.append((previous_val))
        while len(datas)<30:
            datas.append(datas[-1])
    return datas
import numpy as np
parameters = {"Method": ["average", "extreme"], "W": [5, 25], "Pmin": [0.1, 0.2], "Alpha": [0.1, 0.5, 0.9]}
configurations = [dict(zip(parameters, v)) for v in product(*parameters.values())]
ind = 0
data_maks = []
data_means = []
stds = []


c=dict()
c["Method"] = "extreme"
c["W"] = 25
c["Pmin"] = 0.1
c["Alpha"] = 0.5



filenames=[]
data_new = []
data_fromzero = []
data_oneshot = []
data_cont = []

from scipy.stats import friedmanchisquare
ps = []
for pno in range(30):
    problem=SetUnionKnapsack('Data/SUKP',pno)
    filenames.append(problem.dosyaAdi)
    
    file_name = f"Results/cg-CLRL-3-{c['Method']}-{c['Pmin']}-{c['W']}-{c['Alpha']}-0-0-{problem.dosyaAdi}.csv"
    data =get_best_data(file_name, 3)
    data_new.append(data)
    data_fromzero.append(data)

    
    
    file_name = f"Results/cg-CLRL-3-{c['Method']}-{c['Pmin']}-{c['W']}-{c['Alpha']}-0-1-{problem.dosyaAdi}.csv"
    data =get_best_data(file_name, 3)
    data_new.append(data)
    data_oneshot.append(data)
    
    

    file_name = f"Results/cg-CLRL-3-{c['Method']}-{c['Pmin']}-{c['W']}-{c['Alpha']}-0-2-{problem.dosyaAdi}.csv"
    data =get_best_data(file_name, 3)
    data_new.append(data)
    data_cont.append(data)

    ind += 1


#Paper Results
set1 = [6,11,20,28,10,8,3,0,1,5]
set2 = [4,2,15,19,25,22,27,24,12,17]
set3 = [9,7,21,29,18,14,13,16,23,26]


means = np.reshape(np.mean(np.asarray(data_new),axis=1),(30,3))

maxs = np.reshape(np.max(np.asarray(data_new),axis=1),(30,3))

stds = np.reshape(np.std(np.asarray(data_new),axis=1),(30,3))


tablo = np.zeros((10,9))
tablo2 = np.zeros((10,9))
tablo3 = np.zeros((10,9))
for i in range(10):
    tablo[i][0] = np.max(data_fromzero[set1[i]])
    tablo[i][1] = np.mean(data_fromzero[set1[i]])
    tablo[i][2] = np.std(data_fromzero[set1[i]])
    
    tablo[i][3] = np.max(data_oneshot[set1[i]])
    tablo[i][4] = np.mean(data_oneshot[set1[i]])
    tablo[i][5] = np.std(data_oneshot[set1[i]])
    
    tablo[i][6] = np.max(data_cont[set1[i]])
    tablo[i][7] = np.mean(data_cont[set1[i]])
    tablo[i][8] = np.std(data_cont[set1[i]])
    
for i in range(10):
    tablo2[i][0] = np.max(data_fromzero[set2[i]])
    tablo2[i][1] = np.mean(data_fromzero[set2[i]])
    tablo2[i][2] = np.std(data_fromzero[set2[i]])
    
    tablo2[i][3] = np.max(data_oneshot[set2[i]])
    tablo2[i][4] = np.mean(data_oneshot[set2[i]])
    tablo2[i][5] = np.std(data_oneshot[set2[i]])
    
    tablo2[i][6] = np.max(data_cont[set2[i]])
    tablo2[i][7] = np.mean(data_cont[set2[i]])
    tablo2[i][8] = np.std(data_cont[set2[i]])
    
for i in range(10):
    tablo3[i][0] = np.max(data_fromzero[set3[i]])
    tablo3[i][1] = np.mean(data_fromzero[set3[i]])
    tablo3[i][2] = np.std(data_fromzero[set3[i]])
    
    tablo3[i][3] = np.max(data_oneshot[set3[i]])
    tablo3[i][4] = np.mean(data_oneshot[set3[i]])
    tablo3[i][5] = np.std(data_oneshot[set3[i]])
    
    tablo3[i][6] = np.max(data_cont[set3[i]])
    tablo3[i][7] = np.mean(data_cont[set3[i]])
    tablo3[i][8] = np.std(data_cont[set3[i]])
    
from scipy.stats import rankdata
ranks_mean = np.zeros((10,3))
ranks_best = np.zeros((10,3))
for i in range(10):
    ranks_mean[i] = rankdata([-1*tablo3[i][1],-1*tablo3[i][4],-1*tablo3[i][7]],method='dense')
    ranks_best[i] = rankdata([-1*tablo3[i][0],-1*tablo3[i][3],-1*tablo3[i][6]],method='dense')
    a = np.mean(ranks_mean,axis=0)
    b = np.mean(ranks_best,axis=0)

