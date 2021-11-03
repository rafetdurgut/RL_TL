import numpy as np
from os import listdir
from os.path import isfile, join
from copy import deepcopy


class AbstractBaseProblem:
    def objective_function(self):
        raise NotImplemented


class OneMax(AbstractBaseProblem):
    def __init__(self, dimension):
        self.pname = 'OneMax'
        self.dimension = dimension

    def objective_function(self, solution):
        return solution, np.sum(solution)


class ZeroOneKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.dosyaAdi = filenames[fileNo]
        print(self.dosyaAdi)
        self.weights = []
        self.profits = []
        self.qualities = []
        self.non_qualities = []
        with open(f"{folderName}/{self.dosyaAdi}") as f:
            self.dimension, self.capacity = map(int, f.readline().split(' '))
            for i in range(self.dimension):
                line = f.readline().split(' ')
                self.weights.append(float(line[1]))
                self.profits.append(float(line[0]))
                self.qualities.append(float(line[0]) / float(line[1]))
                self.non_qualities.append(float(line[1]) / float(line[0]))

    'If the capacity is not fully filled'

    def optimizing_stage(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.qualities, solution == 0)
        add_index = np.argmax(np.multiply(qualities, solution==0))
        while cap_val + self.weights[add_index] <= self.capacity:
            solution[add_index] = True
            qualities[add_index] = 0
            cap_val += self.weights[add_index]
            add_index = np.argmax(qualities)
        return solution

    'If the capacity is exceed'

    def repair(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        qualities = np.multiply(self.non_qualities, solution)
        while cap_val > self.capacity:
            remove_index = np.argmax(qualities)
            solution[remove_index] = False
            qualities[remove_index] = 0
            cap_val -= self.weights[remove_index]
        return solution

    def objective_function(self, solution):
        cap_val = np.sum(self.weights, where=solution)
        if cap_val > self.capacity:
            solution = self.repair(solution)
            solution = self.optimizing_stage(solution)
        else:
            solution = self.optimizing_stage(solution)

        cap_val = np.sum(self.weights, where=solution)
        #print("cap:"+ str(cap_val))
        sum_val = np.sum(self.profits, where=solution)
        return solution, sum_val

class SetUnionKnapsack(AbstractBaseProblem):
    def __init__(self, folderName, fileNo):
        mypath = folderName
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.dosyaAdi = filenames[fileNo]
        f = open("{}/{}".format(folderName, self.dosyaAdi), "r")
        print("{}/{}".format(folderName, filenames[fileNo]))
        f.readline()
        f.readline()
        line1 = f.readline()
        start = line1.index('=')
        stop = line1.index(' ', start)
        self.m = int(line1[start + 1:stop])
        self.dimension = self.m
        start = line1.index('=', stop)
        stop = line1.index(' ', start)
        self.n = int(line1[start + 1:stop])
        start = line1.index('=', stop)
        iseof = line1.find(' ', start)
        if iseof == -1:
            stop = len(line1) - 1
        else:
            stop = line1.index(' ', start)

        self.C = int(line1[start + 1:stop])

        f.readline()
        f.readline()
        self.p = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.w = list(map(int, f.readline().split()))

        f.readline()
        f.readline()
        self.rmatrix = np.zeros((self.m, self.n), dtype=bool)
        self.items = []
        for i in range(self.m):
            self.items.append([])
            rm = list(map(int, f.readline().split()))
            self.rmatrix[i, :] = deepcopy(rm[:])
            self.items[i] = np.where(self.rmatrix[i][:] == True)
        self.R = np.zeros(self.m)
        self.freqs = np.sum(self.rmatrix, axis=0)
        for i in range(self.m):
            self.R[i] = self.p[i] / np.sum(self.w[j] / self.freqs[j] for j in range(self.n) if self.rmatrix[i][j])

        self.H = np.argsort(self.R)[::-1][:self.m]
        f.close()

    def optimizing_stage(self, solution, temp):
        trial = temp.copy()
        for i in range(self.m):
            a = self.H[i]
            if solution[a] == 0:
                b = self.items[a]
                trial[b] = 1
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    solution[a] = True
                else:
                    trial = temp.copy()

        return solution

    def repair(self, solution):
        temp_sol = np.zeros(self.m, dtype=bool)
        temp = np.zeros((self.n), dtype=bool)
        trial = np.array(temp, copy=True)

        for i in range(self.m):
            a = self.H[i]
            if solution[a]:
                b = self.items[a]
                trial[b] = True
                cap_val = np.sum(self.w, where=trial)
                if cap_val <= self.C:
                    temp[b] = True
                    temp_sol[a] = True
                else:
                    trial = temp.copy()
        return temp_sol, temp

    def objective_function(self, solution):
        temp = np.zeros((self.n), dtype=bool)
        for i in range(self.m):
            if solution[i]:
                temp[self.items[i]] = True
        cap_val = np.sum(self.w, where=temp)

        if cap_val > self.C:
            solution, temp = self.repair(solution)
            solution = self.optimizing_stage(solution, temp)
        else:
            solution = self.optimizing_stage(solution, temp)

        sum_val = np.sum(self.p[i] for i, val in enumerate(solution) if val)
        return solution, sum_val
