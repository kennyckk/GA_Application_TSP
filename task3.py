import numpy as np
import os
import geatpy as ea
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




class Task3(ea.Problem):
    def __init__(self, data):
        name = "test"  # save it for local-->pass to problem init
        self.places = data
        M = 1  # number of objective function
        Dim = self.places.shape[0]  # depend on the clustered data size
        maxormins = [1] * M  # obj function is minimize or max, in this case is to minimize the distance
        varTypes = [1] * Dim  # locations are marked as integer numbers, so 1 for discrete and integer
        lb = [0] * Dim  # smallest idx
        ub = [Dim - 1] * Dim  # biggest idx number
        lbin = [1] * Dim  # include the lower boundary in inputs
        ubin = [1] * Dim  # include upper boundary in inputs

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):  # the process to calculate objective function for the randomly generated chromosomes
        N = x.shape[0]  # to get how many sets of locations

        X = np.hstack([x, x[:, [0]]]).astype(int)  # this is to add the original point for later distance calcuation
        ObjV = []  # array to store N chromosomes' total distance
        for i in range(N):  # loop through each chromosome
            journey = self.places[X[i], 1:3]  # to get the x,y coord data in the order of X[i] i.e. shape(Dim+1,2)
            distance = np.sum(
                np.sqrt(np.sum(np.diff(journey.T) ** 2, axis=0)))  # to calculate the total dist for that chromosome
            ObjV.append(distance)
        f = np.array([ObjV]).T  # to return an array of objective function value for each chromosome
        return f

def run_GA(problem, cluster,dir, maxgen=1000):
    # passing in the problem per cluster defined outside function
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='P', NIND=100),
        # to set the variable to be unique in each chrome, set number of chromosome
        MAXGEN=maxgen,  # maximum generation
        logTras=1)  # to have log for each generation
    algorithm.mutOper.Pm = 0.5  # mutation rate is 0.5%
    saveDirName = dir + str(cluster)
    res = ea.optimize(algorithm, seed=123,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True,
                      dirName=saveDirName)

    return res

def cluster_data(data_name, clusters):
    # add 100 to x coordinate of the copied data and stack them vertically -->(202,2)
    origin=np.loadtxt(os.getcwd()+"/data/"+data_name+".txt", skiprows=1, usecols=(0,1,2)) #inlcuding real location to identify across clusters
    new_add= np.copy(origin)
    new_add[:,1]= new_add[:,1]+100
    new_add[:,0]=new_add[:,0]+origin.shape[0]
    new_data=np.vstack([origin, new_add])

    # use sklearn kmean cluster to identify labels and stack horizontally to column idx 3

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(new_data[:,1:]) # only use xy coord columns 1,2
    label=kmeans.labels_.reshape((new_data.shape[0],1))
    data=np.hstack([new_data,label])

    return data

def print_final(data, cluster,dir):

            x = data[:, 1]
            y = data[:, 2]

            x0 = x[range(len(x))]
            x1 = np.hstack([x[range(1, len(x))], x[0]])
            y0 = y[range(len(y))]
            y1 = np.hstack([y[range(1, len(y))], y[0]])
            xpos = (x0 + x1) / 2
            ypos = (y0 + y1) / 2
            xdir = x1 - x0
            ydir = y1 - y0

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(x, y)
            ax.scatter(x, y, c=data[:, 3])
            ax.plot(x[0],
                    y[0],
                    'o', markersize=12,
                    c='red')

            for i in range(data.shape[0]):
                plt.text(x[i],
                         y[i],
                         int(data[i, 0]),
                         fontsize=10)
                ax.annotate("", xytext=(xpos[i], ypos[i]), xy=(xpos[i] + 0.001 * xdir[i], ypos[i] + 0.001 * ydir[i]),
                            arrowprops=dict(arrowstyle="->", color='k'), size=20)

            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('y')

            plt.savefig(dir+str(cluster)+'/roadmap.svg', dpi=600, bbox_inches='tight')

            plt.show()

def optimize_within_cluster(clusters,data,dir):
    total_route=[] #array to save the idx sequence generated for each cluster
    data_subsets=[] #array to hold the data divided per cluster
    internal_dist=0 # to add on the total distance get from each cluster
    runtime=0 # jsut to keep track of program runtime
    for cluster in range(clusters):
        mask=data[:,-1]==cluster
        clustered_data=data[mask] #to get the data points belong to a specific cluster
        problem=Task3(clustered_data)
        res=run_GA(problem,cluster,dir) #return the result for that cluster

        route_mask=np.hstack([res['Vars'][0, :], res['Vars'][0, 0]])#use the best seq of idx found in that cluster to create a mask
        print_final(clustered_data[route_mask],cluster,dir) #just to print the region route path
        runtime+=res['executeTime']
        internal_dist+=res['ObjV'][0][0]
        total_route.append(route_mask)
        data_subsets.append(clustered_data)
    return total_route, data_subsets, internal_dist, runtime

def optimize_clusters(data, clusters, total_route, data_subsets,
                      internal_dist,runtime,dir):  # to optimize external route between clusters
    regions_data = np.zeros((0, data.shape[1]))
    # create a data set only containing the end point reprentitive of each cluster
    for cluster in range(clusters):
        # to pick the end point on each cluster and create a new region_data only contains them to prepare for next GA run
        cluster_start_idx = total_route[cluster][0]
        cluster_start_loc = data_subsets[cluster][cluster_start_idx]
        regions_data = np.vstack([regions_data, cluster_start_loc])


    sub_problem = Task3(regions_data)  # create another problem for these few cluster representative
    res2 = run_GA(sub_problem, "final", dir,maxgen=50,)
    region_seq = res2['Vars'][0, :]  # the order of clusters found and saved
    total_dist = internal_dist + res2['ObjV'][0][0]  # the total distance for whole trip is internal dist added with external dist of clusters
    runtime+=res2['executeTime']
    # This part is for visualization preparation
    final_route = np.array([])
    for order in region_seq:
        # to get the real order sequence of the all the location
        clus = data_subsets[order]  # get the data belong to that cluster
        seq = total_route[order]  # get the idx order for that particular cluster data
        final_route = np.hstack([final_route, clus[seq, 0]])  # to stack up per cluster the real order of location
    final_route = np.hstack([final_route, final_route[0]])
    print("the best route found:")
    print(final_route)
    print("the total distance is: {}".format(total_dist))
    print("the total runtime is {}".format(runtime))

    final_route_idx = (final_route - 1).astype(int)

    print_final(data[final_route_idx],"final",dir)  # printing the final route function

    return final_route, total_dist

if __name__ =="__main__":
    # main controlling

    clusters = 3  # define how many clusters to exist
    dir='task3/result3_k='
    data = cluster_data("TSPTW_dataset", clusters)  # first load the data and divided by clusters labelled at col idx 3

    total_route, data_subsets, internal_dist, runtime= optimize_within_cluster(clusters,data,dir)  # optimize the route within cluster

    final_route, total_dist = optimize_clusters(data, clusters, total_route, data_subsets, internal_dist, runtime,dir)


