This project is the application of genetic alogrithm on different variants of Travelling Salesman Problem (TSP) with the use of geatpy from https://github.com/geatpy-dev/geatpy
there are 5 tasks as following:
1. Classic TSP
2. Dynamic Optimization TSP
3. Large-scale Optimization TSP
4. Multi-Objective Optimization TSP
5. Time-window Constraint TSP

All tasks made use of the customer xy coordinates data provided under data file. 

Below is the short description of each task:
1. A single objective to find the shortest available round trip route among 100 custmores
2. Same as task 1 but the locations and number of accessible customers would change every 100 generation.
    -2A would reuse the solutions from previous round of optimization and continue optimize for current environment
    -2B would optimize from scratch for every round of changed environment
3. Same as task 1 but with additional 100 customers added and using k-mean clustering to speed up the optimization
4. A multi-objective optimization to a set of solutions of shortest distance and maximized profits also provided in the data
    -4A used MOEA to estimate the pareto optimal fronts
    -4B used weighted Objective funtions to reshape the problem into one single objective
5. Similar to task4 with additional objective to minimize time-violation where visitable timeframe of customers also provided in the data


The code for each task is divided into individual py file e.g task1.py
The code is implemented by python 3.9.12 and is built with below main packages:
geatpy 2.7.0
scikit-learn 1.0.2 (in task3)
numpy 1.21.5
matplotlib 3.5.1

***Please do not change the data folder location.***
