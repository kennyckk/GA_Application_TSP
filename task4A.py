import numpy as np
import os
import geatpy as ea
import matplotlib.pyplot as plt

def load_data(data_name):
    origin=np.loadtxt(os.getcwd()+"/data/"+data_name+".txt", skiprows=1, usecols=(1,2))
    N=origin.shape[0]
    np.random.seed(123)
    profit= np.random.uniform(1,50,(N,1)) #generate a column with randomly generated floats between 1 to 50
    new_data=np.hstack([origin, profit]).astype(float)
    return new_data


class Task4(ea.Problem):
    def __init__(self, data):
        name = "profit and shortest distance"  # save it for local-->pass to problem init
        self.places = data
        M = 2  # number of objective function
        Dim = self.places.shape[0]  # there are 100 places to be ordered; hence 100 "genes" in 1 chromosome
        maxormins = [1, -1]  # first obj is to minimize the distance where 2nd one is to maximize profit
        varTypes = [1] * Dim  # locations are marked as integer numbers, so 1 for discrete and integer
        lb = [0] * Dim  # smallest location number
        ub = [Dim - 1] * Dim  # biggest location number
        lbin = [1] * Dim  # include the lower boundary in inputs
        ubin = [1] * Dim  # include upper boundary in inputs

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):  # the process to calculate objective function for the randomly generated chromosomes

        N = x.shape[0]  # to get how many sets of locations
        X = np.hstack([x, x[:, [0]]]).astype(int)  # this is to add the original point for later distance calcuation

        dist = []  # f1 is the same as all tasks before to calculate shortest distance
        profits = []  # f2 is to calculate the maximum profit for the given sequence in each chromosome
        for i in range(N):  # loop through each chromosome
            journey = self.places[X[i], :2]  # the x y coord are in col 0,1
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, axis=0)))  # to calculate the total dist for that chromosome
            dist.append(distance)

            profit = np.sum(np.absolute(np.diff(self.places[X[i], 2].T)))  # extract the col of the profits according to the generated sequence of location idx
            profits.append(profit)

        f1 = np.array([dist]).T  # f1 array storing distance for each chromosome
        f2 = np.array([profits]).T  # f2 array storing profits for each chromosome
        ObjV = np.hstack([f1, f2])
        return ObjV


def run_GA(problem):
    # passing in the problem per cluster defined outside function
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='P', NIND=100),
        # to set the variable to be unique in each chrome, set number of chromosome
        MAXGEN=1000,  # maximum generation
        logTras=1)  # to have log for each generation

    algorithm.mutOper.Pm = 0.5  # mutation rate is 0.5%
    saveDirName = 'task4/N100G1000result4A'
    res = ea.optimize(algorithm, seed=12,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True,
                      dirName=saveDirName)

    return res

if __name__ == "__main__":
    data = load_data("TSPTW_dataset")
    problem = Task4(data)
    res = run_GA(problem)
