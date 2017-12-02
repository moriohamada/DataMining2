#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Data Mining Task 3: Finding representative examples
@author: Alessandro, Ioana, Morio
Task: find 200 points that are represetnative of the data.
Here, each mapper computes a coreset based on a subsample of the entire dataset.
These coresets are then fed into the reducer, which takes the union of all 
coresets and runs a k-means algorithm (k=200). 
"""

#------------------------------------------------------------------------------
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans

np.random.seed(100)
#------------------------------------------------------------------------------
# Parameters
dim = 250
k = 200
coreset_size = 210
# parameters for k-means in reducer:
num_restarts = 8
max_iter = 10
#------------------------------------------------------------------------------
def mapper(key, value):
    """
    Takes batch of n (3000) data points (each 250 dim) and emits coreset.
    
    First performs bicriteria approx with D^2 sampling procedure, then uses 
    results to obtain coreset of size specified in parameters.
    
    Inputs:
        
        key: None
        
        value: num_examples x 250(dimensions) numpy array
        
    Output:
        
        key: 'key': string to ensure all mappers map to same reducer
        
        value: coreset. (coreset_size x dim) NumPy Array.
    """
    data = value
 
    ### Perform bicriteria approximation with D^2 sampling.
    # Obtain first point, b, uniformly at random:
    b = data[np.random.randint(len(data))]
    # initialize matrix to contain distances from every point in X to all values of b:
    distances = np.empty((0, len(data)))
    
    # Find remaining centres:
    for centre_num in range(1,coreset_size):
        distances = np.vstack((distances, np.linalg.norm(data-b, ord=2, \
                                                         axis=1)**2)) # add distance of every point in data to the most recently sampled center b
        min_distances = distances.min(axis=0) # compute the minimum distance of every point in X to the centers
        prob = min_distances/min_distances.sum()
        b = data[np.random.choice(len(data), replace=False ,p=prob)] # sample point with probability proportional to the minimum distance to the already sampled points

    ### Construct coreset:
    # Parameters for calculation:
    alpha = np.log2(coreset_size) + 1
    C_phi = (1./len(data)) * min_distances.sum()
    
    B_i_ind = distances.argmin(axis=0) # array containing the index of the closest center for each point in 
    # create empty Bi;
    # to contain set of indexes of data points x_i closest to the k_ith centre in B_i[k_i]
    B_i = [[] for centre in range(coreset_size)] 
    for idx, x in enumerate(data):
        B_i[B_i_ind[idx]].append(idx)
        
    # min_B_i = {x in data: i in argmin(d(x_i,B_i))}
    min_B_i = [sum([min_distances[x_i] for x_i in B_i[k_i]]) \
               for k_i in range(coreset_size)]
    # each summed term in sampling dist, q(x); (the long equation in the slides)

    s_term = [2. * alpha * min_B_i[k_i] / (len(B_i[k_i])*C_phi) \
               + 4. * len(data)/len(B_i[k_i]) if B_i[k_i] else 0 \
               for k_i in range(coreset_size)]

    s_x = np.array([alpha * min_distances[x_i] / C_phi + s_term[B_i_ind[x_i]] \
                                                for x_i in range(len(data))])
#    s_x = np.array(
#            [alpha * min_distances[x_i] / C_phi \
#             + [2. * alpha * min_B_i[k_i] / (len(B_i[k_i])*C_phi) \
#             + 4. * len(data) / len(B_i[k_i]) \
#              if B_i[k_i] else 0 for k_i in range(coreset_size)] \
#              for x_i in range(len(data))] )

#             for x_i in range(len(data))] \
#             if B_i[k_i] else 0 for k_i in range(coreset_size)])
    prob_s = s_x/s_x.sum()
    weights = 1./prob_s
    coreset = [(data[s], weights[s]) for s in \
               np.random.choice(len(data), size=coreset_size, replace=False, p=prob_s)]

    yield 'key', coreset
    

def reducer(key, values):
    """
    Performs k-means on coresets fed in from all 9 mappers (reducer only runs
    once).
    
    Runs k-means once or multiple times (specified by num_restarts) and returns
    centers with best score. K is specified at top of script (here, k=200).
 
    Inputs:
        
        key: 'key' (type str)
        
        value: Numpy array ((coreset size * number of mappers run) x 2) of 
        coresets emitted from mappers. Column 0 contains 250-dim coordinates of
        each point in coreset; column 1 contains weights associated with each 
        of these points.            
    """
    coreset = np.array(values[:,0].tolist()) # to get n x d array
    weights = np.array(values[:,1].tolist()) 
    weights = weights/sum(weights) #renormalize
    
    def dist_to_nearest(centers, point):
        """
        Finds distance of point to nearest center
        """
        centers = np.asarray(centers)
        distance,index = KDTree(centers).query(point)
        return index
    
    def weighted_kmeans(coreset, weights, initial_centers, k=200, max_iter=20):
        """
       *** NOTE: right now, this function does nothing. The k-means
       algorithm doesnt make any changes to the initial centers.
       
       There's probably an error here somewhere; or it might just be
       that the coreset samples are already good enough so that k means 
       doesnt change anything. ***
        
        Implements observation-weighted K-means clustering on coresets.
        
        Inputs: 
            coreset: 
                (n x d) Numpy array of data. Each row contains coordinates 
                of point in corset.
            centers: 
                (k x d) NumPy array. Initial cluster centers.  
            k:
                int or None. Number of desired cluster centers. k=200 if None.
            max_iter:
                int or None. max iterations to run k-means algorithm. 
                max_iter = 200 if None.
         
        Returns:
            centers: 
                (k x d) NumPy Array. Each row contains coordinates or returned 
                center found by k-means algorithm
        """
        iter_num = 0
        converged = False
        centers = initial_centers
        ### Run kmeans
        while iter_num < max_iter and not converged:
            iter_num += 1
            updated = 0 #initialize num updated centers
            #assign each point to closest cluster
            assigned_center_ind = []
            for point in coreset: # for each data point
                distances= [] # initialize list of distances
                nearest_center_idx = dist_to_nearest(centers, point)
                assigned_center_ind.append(nearest_center_idx) # list contain cluster index that each data point is closest to
                
#            print 'assigned_center_ind', assigned_center_ind
            # find new centers:
            for center_idx, center in enumerate(centers):
                points_in_cluster = np.array([point_idx for point_idx, p \
                                              in enumerate(coreset) if \
                                              assigned_center_ind[point_idx]\
                                              == center_idx])
                

                if len(points_in_cluster)>0: # if at least one point in cluster
#                    print 'center and weight shape', coreset[points_in_cluster].shape,weights[points_in_cluster].shape
##                    print 'center[0]', center[0]
##                    print coreset[points_in_cluster,0]
##                    print weights[points_in_cluster]
#                    print (np.average(coreset[points_in_cluster], \
#                                                     axis = 0, \
#                                                     weights = \
#                                                     weights[points_in_cluster]))[0], 'is what is should be'
#                    
                    centers[center_idx] = np.average(coreset[points_in_cluster], \
                                                     axis = 0,
                                                     weights = \
                                                     weights[points_in_cluster])

                    if (centers[center_idx] != center).all():
                        updated+=1
            print 'updated', updated, 'centers'
            if updated==0: 
                converged = True
                print 'converged on iteration', iter_num
       
            print 'iter num', iter_num
        
        # find quantization error: 
        score = np.sum([dist_to_nearest(centers, x_i) ** 2 
                         for idx, x_i in enumerate(coreset)])
        
        return centers, score
    
    yielded_centers = np.zeros((num_restarts, k, dim))
    scores = np.zeros(num_restarts)
    
    for run in range(num_restarts): # re-initialize k-means several times
        inv_weights = [1/w for w in weights]
        inv_weights = inv_weights/sum(inv_weights)
        init_centersidx = [np.random.choice(len(coreset),size=k,replace=False,p=inv_weights)] # weight-biased sampling
        # init_centersidx = [np.random.choice(len(coreset),size=k,replace=False)] # or uniform sampling
        init_centers = np.array(coreset[init_centersidx].tolist())
        print init_centers.shape, 'are initial center shapes'
        yielded_centers[run], scores[run] = weighted_kmeans(coreset=coreset, \
                                                            weights=weights, \
                                                            initial_centers=init_centers)
    
    # find best centers
    best_centers = yielded_centers[scores.argmin()]
    
    print 'scores were', scores
    
    yield best_centers
    
            
        
        
    
        
    
    
