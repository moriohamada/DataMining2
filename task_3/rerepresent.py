#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Data Mining Task 3: Finding representative examples

@author: Alessandro, Ioana, Morio

Task: find 200 points that are represetnative of the data.

Here, each mapper computes a coreset based on a subsample of the entire dataset.
These coresets are then fed into the reducer, which takes the union of all 
coresets and again produces a coreset of 200 microclusters. 
"""

#------------------------------------------------------------------------------
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans
#------------------------------------------------------------------------------
# Parameters
dim = 250
k = 200
coreset_size = 300
# parameters for k-means in reducer:
num_restarts = 4
max_iter = 20
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
    s_x = np.array(
            [alpha * min_distances[x_i] / C_phi \
           + [2. * alpha * min_B_i[k_i] / (len(B_i[k_i])*C_phi) \
              + 4. * len(data) / len(B_i[k_i]) \
              if B_i[k_i] else 0 for k_i in range(coreset_size)][B_i_ind[x_i]] \
              for x_i in range(len(data))] )

    prob_s = s_x/s_x.sum()
    weights = 1./(coreset_size * prob_s)
    coreset = [(data[s], weights[s]) for s in \
               np.random.choice(len(data), size=coreset_size, replace=False, p=prob_s)]

    yield 'key', coreset   

def reducer(key, values):
    """
    Takes union of coresets, and re-produces another coreset with k=200 points.
    
    Runs k-means once or multiple times (specified by num_restarts) and returns
    centers with best score. K is specified at top of script (here, k=200).
 
    Inputs:
        
        key: 'key' (type str)
        
        value: Numpy array ((coreset size * number of mappers run) x 2) of 
        coresets emitted from mappers. Column 0 contains 250-dim coordinates of
        each point in coreset; column 1 contains weights associated with each 
        of these points.            
        
    """
    data = np.array(values[:,0].tolist()) # to get n x 250 array
    weights = values[:,1]
    weights = weights / sum(weights) # re-normalize
    
    ### Perform bicriteria approximation with D^2 sampling.

    # Obtain first point according to weight
    b = data[np.random.choice(range(len(data)),p=list(weights))]
    distances = np.empty((0, len(data)))
    # Find remaining centres:
    for centre_num in range(1,k):
        distances = np.vstack((distances, np.linalg.norm(data-b, ord=2, \
                                                         axis=1)**2)) # add distance of every point in X to the new sampled point b
        min_distances = distances.min(axis=0) # compute the current minimum distance of every point in X to the sampled points
        prob = min_distances * weights / sum(min_distances * weights)
        b = data[np.random.choice(len(data), replace=False, p=list(prob))] # sample point with probability proportional to the minimum distance to the already sampled points

    # Construct coreset:
    # parameters for calculation:
    alpha = np.log2(k) + 1
    C_phi = (1./len(data)) * min_distances.sum()
    
    B_i_ind = distances.argmin(axis=0) # array containing the index of the closest center for each point in 
    # create empty Bi;
    # to contain set of indexes of data points x_i closest to the k_ith centre in B_i[k_i]
    B_i = [[] for centre in range(k)] 
    for idx, x in enumerate(data):
        B_i[B_i_ind[idx]].append(idx)
    
    # min_B_i = {x in data: i in argmin(d(x_i,B_i))}
    min_B_i = [sum([min_distances[x_i] for x_i in B_i[k_i]]) \
               for k_i in range(k)]
    
    # each summed term in sampling dist, q(x):
    s_x = np.array(
            [alpha * min_distances[x_i] / C_phi \
           + [2. * alpha * min_B_i[k_i] / (len(B_i[k_i])*C_phi) \
              + 4. * len(data) / len(B_i[k_i]) \
              if B_i[k_i] else 0 for k_i in range(k)][B_i_ind[x_i]] \
              for x_i in range(len(data))] )

    prob_s = s_x * weights/sum(s_x * weights)
    weights = 1./(k * prob_s)
    centers = np.empty((num_restarts,k,250))
    
    for run in range(num_restarts):
        centers[run] = [data[s] for s in np.random.choice(len(data), size=k, \
                   replace=False,p=list(prob_s))]
        
    def dist_to_nearest(centers, point):
        """
        Finds distance of point to nearest center
        """
        centers = np.asarray(centers)
        distance,index = KDTree(centers).query(point)
        return distance
    
    scores = []
    for run in range(num_restarts):
        score = np.sum([dist_to_nearest(centers[run], x_i) ** 2 \
                         for idx, x_i in enumerate(data)])
        scores.append(score)
        
    print 'scores:', scores
    best_centers = centers[np.array(scores).argmin()]
        

    yield best_centers
    
    
    
