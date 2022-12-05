import numpy as np
import os
import geatpy as ea
import matplotlib.pyplot as plt


class Task1(ea.Problem):
    def __init__(self, data_name):
        name = data_name  # save it for local-->pass to problem init
        self.places = np.loadtxt(os.getcwd() + "/data/" + data_name + ".txt", skiprows=1, usecols=(1, 2))
        M = 1  # number of objective function
        Dim = self.places.shape[0]  # there are 100 places to be ordered; hence 100 "genes" in 1 chromosome
        maxormins = [1] * M  # obj function is minimize or max, in this case is to minimize the distance
        varTypes = [1] * Dim  # locations are marked as integer numbers, so 1 for discrete and integer
        lb = [0] * Dim  # smallest location number
        ub = [Dim - 1] * Dim  # biggest location number
        lbin = [1] * Dim  # include the lower boundary in inputs
        ubin = [1] * Dim  # include upper boundary in inputs

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):  # the process to calculate objective function for the randomly generated chromosomes
        N = x.shape[0]  # to get how many sets of locations

        X = np.hstack([x, x[:, [0]]]).astype(int)  # this is to add the original point for later distance calcuation
        ObjV = []  # array to store N chromosomes' total distance
        for i in range(N):  # loop through each chromosome
            journey = self.places[X[i], :]  # to get the x,y coord data in the order of X[i] i.e. shape(Dim+1,2)
            distance = np.sum(
                np.sqrt(np.sum(np.diff(journey.T) ** 2, axis=0)))  # to calculate the total dist for that chromosome
            ObjV.append(distance)
        f = np.array([ObjV]).T  # to return an array of objective function value for each chromosome
        return f

def printRoute(res,problem,saveDirName):
    if res['success']:
        print('shortest distance：%s' % res['ObjV'][0][0])
        print('best route found：')
        best_journey = np.hstack([res['Vars'][0, :], res['Vars'][0, 0]])
        for i in range(len(best_journey)):
            print(int(best_journey[i]), end=' ')
        print()
        start = best_journey[0].astype(int)
        x = problem.places[best_journey.astype(int), 0]
        y = problem.places[best_journey.astype(int), 1]

        x0 = x[range(len(x))]
        x1 = np.hstack([x[range(1, len(x))], problem.places[start, 0]])
        y0 = y[range(len(y))]
        y1 = np.hstack([y[range(1, len(y))], problem.places[start, 1]])
        xpos = (x0 + x1) / 2
        ypos = (y0 + y1) / 2
        xdir = x1 - x0
        ydir = y1 - y0

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(x, y)

        ax.plot(problem.places[start, 0],
                problem.places[start, 1],
                'o', markersize=12,
                c='red')

        for i in range(len(best_journey)):
            plt.text(problem.places[int(best_journey[i]), 0],
                     problem.places[int(best_journey[i]), 1],
                     int(best_journey[i]),
                     fontsize=10)
            ax.annotate("", xytext=(xpos[i], ypos[i]), xy=(xpos[i] + 0.001 * xdir[i], ypos[i] + 0.001 * ydir[i]),
                        arrowprops=dict(arrowstyle="->", color='k'), size=20)

        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(saveDirName + '/roadmap.svg', dpi=600, bbox_inches='tight')
        plt.show()
    else:
        print('no solution')


if __name__== "__main__":
    # to initialize the class of task1 with all var definition
    problem = Task1("TSPTW_dataset")
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='P', NIND=100),
        # to set the variable to be unique in each chrome, set number of chromosome
        MAXGEN=2000,  # maximum generation
        logTras=1)  # to have log for each generation
    algorithm.mutOper.Pm = 0.9 # mutation rate is 0.5%
    algorithm.recOper.XOVR =0.7 #Selection rate is
    saveDirName = 'task1/mutate=0.9'
    res = ea.optimize(algorithm, seed=123,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True,
                      dirName=saveDirName)
    printRoute(res, problem,saveDirName)
