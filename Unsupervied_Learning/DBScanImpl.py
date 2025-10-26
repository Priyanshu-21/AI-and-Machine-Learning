# Implementation of DBScan Query Based algorithm 
import numpy as np 
from sklearn.neighbors import KDTree

class DBScan: 
    def __init__(self, x):
        self.x = x

    def DBScan_query_based(x, epl, minPts):
        n = x.shape[0]
        labels = np.full(n, -1)
        cluster_id = 0

        tree = KDTree(x)
        
        for i in range(n):
            
            if (labels[i] != -1):
                continue
            
            neighbours = tree.query_radius(x[i].reshape(1, -1), r= epl)[0]
            
            if (len(neighbours) < minPts):
                #Noise in clusters 
                labels[i] = -2 

            else:
                cluster_id += 1
                labels[i] = cluster_id

                seeds = list(neighbours)
                while (seeds):
                    current_point = seeds.pop()
                    
                    if(labels[current_point]) == -2:
                        labels[current_point] = cluster_id
                    
                    if(labels[current_point]) == -1:
                        continue

                    labels[current_point] = cluster_id

                    #check for neighbour points of cluster in radius 
                    new_neighbours = tree.query_radius(x[current_point].reshape(1, -1), r= epl)[0]

                    if(len(new_neighbours) >= minPts):
                        #Neighbours 
                        seeds.extend(new_neighbours)

        return labels

