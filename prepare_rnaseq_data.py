# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:12:54 2023

@author: youm
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture as GMM
import numpy as np

#PART ONE
#'''

dist = pd.read_csv("mex_matrix_filtered_distal.csv").drop(["feature_id","feature_type"],axis=1)
adj = pd.read_csv("mex_matrix_filtered_adjacent.csv").drop(["feature_id","feature_type"],axis=1)

print(dist)

dl = ["location"] + list(np.full(dist.shape[1]-1,"distal",dtype=object))
print(len(dl))

dist.loc[len(dist.index)] = dl
adj.loc[len(dist.index)] =  ["location"] + list(np.full(adj.shape[1]-1,"adjacent",dtype=object))

print(dist,'dist')
print(adj,'adj')
print(list(dist["gene"])[:10],"first 10 gene column")

dist.index = list(dist["gene"])
adj.index = list(adj["gene"])
dist = dist.drop(["gene"],axis=1)
adj = adj.drop(["gene"],axis=1)
df = pd.concat([dist,adj],axis=1)
print(dist.shape,adj.shape,df.shape)



key = "DEPRECATED" !=  pd.Series(df.index,index=df.index).apply(lambda n: str(n).split('_')[0])   #.apply(lambda n: n.split('_')[0])
#key = []
#for lab in df.loc[:,"gene"]:
print(key.sum())
print(key)

df = df.loc[key,:]
df.to_csv("rna seq rawish data.csv")
print(df)
countsDF = df.iloc[:-1,1:]

print(countsDF,'cdf')
print(countsDF.values,"cdf values")


def kmeans(df9):
    ch = 'km'
    print(df9,"df in kmeans")
    ncl = 30
    if ch == "km":
        km = KMeans(n_clusters=ncl)
        km.fit(df9)
        df9["kmeans"] = km.labels_
    else:
        gmm = GMM(n_components=ncl).fit(df9)
        df9["GMM"] = gmm.predict(df9)
    return(df9)




labDF = kmeans(countsDF)

madf = pd.concat([pd.Series(labDF.index,index=labDF.index),labDF["GMM"]],axis=1)
madf.to_csv("gene km clustering1.csv")
#'''
#PART TWO



