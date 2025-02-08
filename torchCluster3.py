# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:15:00 2023

@author: youm
"""

import torch
import torchClusterTester1 as tCT
import pandas as pd
import warnings
import time
#warnings.filterwarnings("ignore")#, category=FutureWarning)

LR = 10e-6




def main(df,ncl,centroids = None,MAXI = 10):
    if type(centroids) == type(None):
        centroids = initialize(ncl,df.shape[1])
    #dT = makeDataTensor(df,ncl)
    centroids,mInds = solve(df,centroids,MAXI = MAXI)
    #print(centroids,'centroids')
    #print(mInds.indices,'indices')
    return(list(mInds.indices.detach().numpy()),centroids)
    
def solve(df,cents,MAXI = 10):
    t0 = time.time()
    optim = torch.optim.SGD([cents], lr=LR)
    dA = torch.tensor(df.values).type(torch.float).reshape(df.shape[0],1,df.shape[1])
    SK = 1
    for i in range(MAXI):
        t1 = time.time()
        print((t1-t0)/60,'m, loop')
        if i < SK:
            print('cents in solve',cents.size())
        
        #print(cents,'cents in solve',cents.size())
        distances = torch.subtract(dA,cents)
        t2 = time.time()
        print((t2-t1)/60,'subtract')
        #print(distances,'distances',distances.size())
        
        if i < SK:
            print('\n\n\n',dA,'input\n\n',cents,'cents\n\n',distances,'distances',distances.size())
        #to push algorithm towards finding an equal number in each centroid, 
        #make it so loss of each cluster is calculated seperately and squared/cubed
        distances = distances**2
        t25 = time.time()
        print((t25-t2)/60,'square')
        distances = torch.sum(distances,axis=-1)
        t3 = time.time()
        print((t3-t25)/60,'sum')
        if i == SK:
            print(distances,distances.size(),'distances after sq n sum')
        mInds = torch.min(distances,axis=-1)
        if i < SK:
            print(mInds,'mInds',mInds.values.size()) #minds.indices not used unless implementing loss of each cluster
        
        loss = torch.sum(mInds.values)
        t4 = time.time()
        print((t4-t3)/60,'getLoss')
        print(float(loss.detach()),'          loss')
        if True:#i < MAXI - 1:
            loss.backward()
            optim.step()
        t5 = time.time()
        print((t5-t4)/60,'backprop')
    return(cents,mInds)
        
 
def getDistances(df,cents):
    
    return()
    
'''    
def makeDataTensor(df,ncl):
    dT = []
    for i in range(df.shape[0]):
        dA = []
        for j in range(ncl):
            dA.append(df.values[i,:])
        dT.append(dA)
    dT = torch.tensor(dT)
    print(dT.size(),'data tensor size')
    return(dT)
'''

    
def initialize(height,width):
    #cents = torch.ones((1,height,width)) - .5
    cents = torch.rand((1,height,width)) - .5
    #cents = torch.reshape(cents,(1,cents.size()[0],df.shape[1]))
    cents =  torch.tensor(cents.detach(),requires_grad=True)
    print(cents,'initialize cents',cents.size())
    return(cents)

if __name__ == "__main__":   
    dit = 0
    
    if dit:     
        df = pd.DataFrame([[1,1,1,1,1],[1,2,3,4,5]])
        print(df,'df in')
        main(df,3)
    else:
        tCT.main()
        '''
        # Given input tensors
        points = torch.tensor([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
        centroids = torch.tensor([[-0.2466, 0.3791, -0.2941, 0.1369, 0.0113],
                                  [0.3328, -0.2321, -0.2723, 0.2966, 0.1944],
                                  [0.2529, 0.4797, -0.0178, -0.2064, -0.3225]])
        
        # Reshape centroids for broadcasting
        centroids_reshaped = centroids.reshape(1, 3, 5)
        print(points.size(),centroids.size())
        
        # Calculate the difference matrix
        difference_matrix = points - centroids_reshaped
        
        # Print the result
        print(difference_matrix)
        '''

'''
import numpy as np

# Define the points and centroids matrices
points = np.array([[1, 2, 3], [4, 5, 6]])
centroids = np.array([[1, 2], [1, 2], [1, 2]])
print(centroids,'centroids')
# Reshape centroids for broadcasting
centroids_reshaped = centroids.T.reshape(2, 1, 3)
print(centroids,'centroids')

# Calculate the difference matrix
difference_matrix = points - centroids_reshaped

# Print the result
print(difference_matrix) 
'''