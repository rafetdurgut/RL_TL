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
        previous_iter = 0
        previous_val = 0
        for row in csv_reader:
            if len(row) > 0:
                iteration, val = row
                if iteration < previous_iter:
                    datas.append((previous_val))
                else:
                    previous_val = val
                previous_iter = iteration
        datas.append((previous_val))
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
c["Alpha"] = 0.1

c_2=dict()
c_2['Method']='average'
c_2['Pmin']=0.2
c_2['W']=5
c_2['Alpha']=0.9


filenames=[]
data_new = []
data_ref = []
data_rand = []
from scipy.stats import friedmanchisquare
ps = []
for pno in range(30):
    problem=SetUnionKnapsack('Data/SUKP',pno)
    filenames.append(problem.dosyaAdi)
    
    file_name = f"Results/cg-CLRL-3-{c['Method']}-{c['Pmin']}-{c['W']}-{c['Alpha']}-0.3-{pno}.csv"
    data =get_best_data(file_name, 3)
    data_new.append(data)
    
    
    file_name = f"Analysis/cg-PM-3-{c_2['Method']}-{c_2['Pmin']}-{c_2['W']}-{c_2['Alpha']}-{pno}.csv"
    data =get_best_data(file_name, 3)
    data_ref.append(data)
    
    
    file_name = f"Analysis/cg-RND-5-extreme-0.1-25-0.1-{pno}.csv"
    data =get_best_data(file_name, 3)
    data_rand.append(data)
    
    ind += 1

ps = []


#Paper Results
set1 = [0,1,8,9,14,15,20,21,26,27]
set2 = [28,29,4,5,10,11,16,17,22,23]
set3 = [2,3,6,7,12,13,18,19,24,25]

mean_1 = np.mean(data_new,axis=1)
mean_2 = np.mean(data_ref,axis=1)
mean_3 = np.mean(data_rand,axis=1)
w, p = friedmanchisquare(mean_1[set1],mean_2[set1],mean_3[set1])
ps.append(p)

w, p = friedmanchisquare(mean_1[set2],mean_2[set2],mean_3[set2])
ps.append(p)

w, p = friedmanchisquare(mean_1[set3],mean_2[set3],mean_3[set3])
ps.append(p)

mean_1 = np.max(data_new,axis=1)
mean_2 = np.max(data_ref,axis=1)
mean_3 = np.max(data_rand,axis=1)
w, p = friedmanchisquare(mean_1[set1],mean_2[set1],mean_3[set1])
ps.append(p)

w, p = friedmanchisquare(mean_1[set2],mean_2[set2],mean_3[set2])
ps.append(p)

w, p = friedmanchisquare(mean_1[set3],mean_2[set3],mean_3[set3])
ps.append(p)


tablo = np.zeros((10,9))
tablo2 = np.zeros((10,9))
tablo3 = np.zeros((10,9))
for i in range(10):
    tablo[i][0] = np.max(data_new[set1[i]])
    tablo[i][1] = np.mean(data_new[set1[i]])
    tablo[i][2] = np.std(data_new[set1[i]])
    
    tablo[i][3] = np.max(data_ref[set1[i]])
    tablo[i][4] = np.mean(data_ref[set1[i]])
    tablo[i][5] = np.std(data_ref[set1[i]])
    
    tablo[i][6] = np.max(data_rand[set1[i]])
    tablo[i][7] = np.mean(data_rand[set1[i]])
    tablo[i][8] = np.std(data_rand[set1[i]])
    
for i in range(10):
    tablo2[i][0] = np.max(data_new[set2[i]])
    tablo2[i][1] = np.mean(data_new[set2[i]])
    tablo2[i][2] = np.std(data_new[set2[i]])
    
    tablo2[i][3] = np.max(data_ref[set2[i]])
    tablo2[i][4] = np.mean(data_ref[set2[i]])
    tablo2[i][5] = np.std(data_ref[set2[i]])
    
    tablo2[i][6] = np.max(data_rand[set2[i]])
    tablo2[i][7] = np.mean(data_rand[set2[i]])
    tablo2[i][8] = np.std(data_rand[set2[i]])
for i in range(10):
    tablo3[i][0] = np.max(data_new[set3[i]])
    tablo3[i][1] = np.mean(data_new[set3[i]])
    tablo3[i][2] = np.std(data_new[set3[i]])
    
    tablo3[i][3] = np.max(data_ref[set3[i]])
    tablo3[i][4] = np.mean(data_ref[set3[i]])
    tablo3[i][5] = np.std(data_ref[set3[i]])
    
    tablo3[i][6] = np.max(data_rand[set3[i]])
    tablo3[i][7] = np.mean(data_rand[set3[i]])
    tablo3[i][8] = np.std(data_rand[set3[i]])
    
from scipy.stats import rankdata
ranks_mean = np.zeros((10,3))
ranks_best = np.zeros((10,3))
for i in range(10):
    ranks_mean[i] = rankdata([-1*tablo[i][1],-1*tablo[i][4],-1*tablo[i][7]],method='dense')
    ranks_best[i] = rankdata([-1*tablo[i][0],-1*tablo[i][3],-1*tablo[i][6]],method='dense')
    a = np.mean(ranks_mean,axis=0)
    b = np.mean(ranks_best,axis=0)