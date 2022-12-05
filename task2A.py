import numpy as np
import os
import geatpy as ea
import matplotlib.pyplot as plt
import math


class Task2(ea.Problem):
    def __init__(self, data):
        name = "task2A"  # save it for local-->pass to problem init
        self.places = data

        M = 1  # number of objective function
        Dim = self.places.shape[0]  # there are 100 places to be ordered; hence 100 "genes" in 1 chromosome
        maxormins = [1] * M  # obj function is minimize or max, in this case is to minimize the distance
        varTypes = [1] * Dim  # locations are marked as integer numbers, so 1 for discrete and integer

        lb = [0] * Dim  # lowerbound for first 50+10e will be 0 ; 50+10e for the rest in the chromosome
        ub = [Dim - 1] * Dim  # upbound for first 50+10e will be 50+10e-1; dim-1 for the rest in the chromosome
        lbin = [1] * Dim  # include the lower boundary in inputs
        ubin = [1] * Dim  # include upper boundary in inputs

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):  # the process to calculate objective function for the randomly generated chromosomes
        N = x.shape[0]  # to get number of sets of locations order

        X = np.hstack([x, x[:, [0]]]).astype(int)  # this is to add the original point for later distance calcuation
        ObjV = []  # array to store N chromosomes' total distance
        for i in range(N):  # loop through each chromosome
            journey = self.places[X[i],
                      :]  # to get the x,y coord data in the order of X[i] i.e. shape(Dim+1,2), only xy coordinate extract
            distance = np.sum(
                np.sqrt(np.sum(np.diff(journey.T) ** 2, axis=0)))  # to calculate the total dist for that chromosome
            ObjV.append(distance)
        f = np.array([ObjV]).T  # to return an array of objective function value for each chromosome

        return f

def extend_lastbest(res,e):
    front=res['Vars'][0, :]
    next_env=50+10*(e+1)
    if e==4: #boundary condition to include the 101th datapoint
        next_env+=1
    rear=np.arange(50+10*e,next_env,1);np.random.shuffle(rear) #to get the portion that meet the next environment
    lastbest=np.hstack([front,rear])
    return lastbest

def printRoute(res):
    if res['success']:
            print('shortest distance：%s' % res['ObjV'][0][0])
            print('best route found：')
            best_journey = np.hstack([res['Vars'][0, :], res['Vars'][0, 0]])
            for i in range(len(best_journey)):
                print(int(best_journey[i]), end=' ')
            print()
            start=best_journey[0].astype(int)
            x=problem.places[best_journey.astype(int), 0]
            y=problem.places[best_journey.astype(int), 1]

            x0 = x[range(len(x))]
            x1 = np.hstack([x[range(1,len(x))],problem.places[start, 0]])
            y0 = y[range(len(y))]
            y1 = np.hstack([y[range(1,len(y))],problem.places[start, 1]])
            xpos = (x0+x1)/2
            ypos = (y0+y1)/2
            xdir = x1-x0
            ydir = y1-y0

            fig, ax= plt.subplots()
            ax.plot(x,y)
            ax.scatter(x,y)

            ax.plot(problem.places[start, 0],
                     problem.places[start, 1],
                     'o',markersize=12,
                     c='red')

            for i in range(len(best_journey)):
                plt.text(problem.places[int(best_journey[i]), 0],
                         problem.places[int(best_journey[i]), 1],
                         int(best_journey[i]),
                         fontsize=10)
                ax.annotate("", xytext=(xpos[i],ypos[i]),xy=(xpos[i]+0.001*xdir[i],ypos[i]+0.001*ydir[i]),arrowprops=dict(arrowstyle="->", color='k'), size = 20)



            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(saveDirName + '/roadmap.svg', dpi=600, bbox_inches='tight')
            plt.show()
    else:
            print('no solution')

def load_data(name,e):
    places = np.loadtxt(os.getcwd() + "/data/" + name + ".txt", skiprows=1, usecols=(1, 2))
    ecap = 50 + 10 * e  # the constraint of number of acessible locations from datasets
    if e <= 4:
        places = places[:ecap]

    places[:, 0] = places[:, 0] + 2 * e * math.cos(e * math.pi / 2)
    places[:, 1] = places[:, 1] + 2 * e * math.sin(e * math.pi / 2)

    return places

if __name__=="__main__":
    np.random.seed(123) #make result reproducible
    lastbest = None #the first environment will have no last best route found from last env
    shortest_d=[]
    time=[]
    for e in range(6):

        data=load_data("TSPTW_dataset",e) #preprocess the data to correponding environement
        # problem defining
        problem = Task2(data) #need to pass the env to prob definition for varting datasize & xy coord
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='P', NIND=900),# to set the variable to be unique in each chrome, set number of chromosome
            MAXGEN=100,  # maximum generation for each e as per task2 defined
            logTras=1)  # to have log for each generation
        algorithm.mutOper.Pm = 0.5  # mutation rate is 0.5%
        saveDirName = 'task2A500/result_e=' + str(e)
        res = ea.optimize(algorithm,
                          seed=123,
                          verbose=True,
                          drawing=1,
                          outputMsg=True,
                          drawLog=False,
                          saveFlag=True,
                          dirName=saveDirName,
                          prophet=lastbest)
        printRoute(res)
        shortest_d.append(res['ObjV'][0,0])
        time.append(res['executeTime'])

        if e > 4: #e=5 will not have next environment
            continue
        lastbest = extend_lastbest(res, e)
    print(shortest_d)
    print(time)