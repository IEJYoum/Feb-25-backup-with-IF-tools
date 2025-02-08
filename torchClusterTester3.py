# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:26:01 2023

@author: youm
"""

#srun --pty --time=1-0 --mem=64G --gres=gpu:1 --partition=gpu bash -i
# Y:\  ==  /home/groups/graylab_share/Chin_Lab/ChinData

import pandas as pd
import torchCluster4 as tC
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore


TSTEM = 'iy_hta14'
ITERS = 50
MAXI = 2
NCL = 10

def main():
    '''
    df = pd.read_csv('rna seq rawish data.csv',index_col=0).iloc[:-1,:].astype(float)
    print(df)
    clusters,centroids = tC.main(df,10,MAXI=20)
    groups = pd.DataFrame(data=clusters,index = df.index).transpose()
    groups.to_csv('torchcluster_gene_classes.csv')
    '''
    df,obs,dfxy = preload(9,9,9)
    oxy = dfxy.copy()
    df = df.apply(zscore)
    print(obs.columns)
    colors = ["red","g","b","#FFA000","lightgray",'magenta','darkgreen','pink','cyan','tan','yellow','darkred']
    #spatialLite(obs,obs["slide_scene"].iloc[0],colors,dfxy,obs['Primary Celltype autoCellType res: 1.0'].unique(),3)
    clusters,centroids = tC.main(df,NCL,MAXI=MAXI)
    #print(clusters,'clusters')
    obs["cluster"] = clusters
    for i in range(ITERS):
        spatialLite(obs,obs["slide_scene"].iloc[0],colors,oxy,obs['cluster'].unique(),-1)
        clusters,centroids = tC.main(df,NCL,centroids=centroids,MAXI=MAXI) #dfxy.apply(zscore) must change above also #centroids must be tensor
        obs["cluster"] = clusters
    spatialLite(obs,obs["slide_scene"].iloc[0],colors,oxy,obs['cluster'].unique(),-1)
    #'''

def spatialLite(nobs,scene,colors,nxy,uch,ch1,ymin=0):
        key=nobs["slide_scene"]==scene
        sobs = nobs.loc[key,:]
        #sdf = ndf.loc[key,:]
        sxy = nxy.loc[key,:]
        #ax.set_aspect('equal')
        #ax.legend(uch,colors,bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500))
            #print(max(sxy.iloc[:,1]),"max Y")
        except Exception as e:
            print(e,"error setting fig and ax",scene)
            print(sxy.isna().any(),"isna")
            fig,ax = plt.subplots()
        for i,ty in enumerate(uch):
            co = colors[i]
            #print(sobs.columns[ch1])
            key1 = sobs[sobs.columns[ch1]]==ty
            #print(key1)
            #tobs = sobs.loc[key,:]
            #tdf = sdf.loc[key,:]
            txy = sxy.loc[key1,:]
            x = []
            y = []
            if txy.shape[0] == 0:
                continue
            for j in range(txy.shape[0]):
                pt = list(txy.iloc[j,:])
                #coords.append((pt[0],pt[1]))
                x.append(pt[0])
                y.append(-pt[1])
            Y = pd.Series(y)
            #sxy = list(sxy.astype(float))
            #print(sxy)
            Y += max(sxy.iloc[:,1])+ymin
            ax.scatter(x,Y,color=co,label=ty,s=1.2)
        lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, scatterpoints=1, fontsize=10)
        try:
            for k in range(len(uch)):
                lg.legendHandles[k]._sizes = [30]
        except Exception as e:
            print(e,"index k:",k,len(uch),lg.legendHandles)
        plt.title(scene+" "+nobs.columns[ch1])
        plt.show()



def preload(bl1,bl2,bl3,path = ''):
    if path == "none" or path == "":
        path = os.getcwd()
    print(path)
    for file in os.listdir(path):
        if TSTEM == "_".join(file.split("_")[:-1]):
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0)
    obs.name = obs.columns[-1]
    obs = obs.loc[list(df.index),:].astype(str)
    dfxy = dfxy.loc[list(df.index),:]
    print(all(obs.index==df.index),"all index the same")
    print(df.index,obs.index)
    return(df,obs,dfxy)

if __name__ == "__main__":
    main()
