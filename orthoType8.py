# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:51:43 2023

@author: youm
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
import os
import seaborn as sns
import allcolors as allc


NITERS = 10**8
MIN = 10

COL = 'slide'


def train(df,obs,dfxy=None,sep=COL):
    newCols = []
    for biom in df.columns:
        if "nuclei_" in biom:
            newCols.append(biom)
        else:
            nn = biom.split("_")[0]+"_"
            if nn in newCols:
                nind = newCols.index(nn)
                newCols[nind] = df.columns[nind]
                newCols.append(biom)
            else:
                newCols.append(nn)
    df = pd.DataFrame(data=df.values,index=df.index,columns = newCols)
    if not sep or sep not in obs.columns:
        ch,uch = obMenu(obs,"category whose subcategories will be individually annotated (e.g. batch or patient)")
        sep = obs.columns[ch]

    obs["orthoThresh phenotype " + sep] = ''
    for i,uo in enumerate(sorted(list(obs.loc[:,sep].unique()))):

        print(uo)
        key = obs.loc[:,sep] == uo
        tdf = df.loc[key,:].apply(zscore)
        mins = tdf.min(axis=0)
        tdf -= mins
        tdf += 1
        #tdf = np.log(tdf)
        tdf = tdf ** 4

        #quant = tdf.quantile(.97,axis=0)
        #tdf = tdf.clip(upper=quant,axis=1)


        #return(1,2,3)
        tobs = obs.loc[key,:]

        cm = corMat(tdf)
        #showHeatmap(cm)
        #getThresh(cm,df,obs)
        thresho = getThresh(cm,tdf,tobs,t2 = uo)
        for i,biom in enumerate(df.columns):
            bn = biom.split('_')[0]+'_ '
            keyt = tdf.loc[:,biom] > thresho[i]
            tobs.loc[keyt,"orthoThresh phenotype " + sep] += bn
        obs.loc[key,:] = tobs
    return(df,obs,dfxy)



def getLoss(biom,abiom,thresh):
    sbiom = (biom - thresh) * 10
    sbiom = 2 * torch.sigmoid(sbiom) - 1
    pos = 2*( torch.sigmoid((sbiom+1) * 10) - .5)
    neg = 2*( torch.sigmoid((sbiom-1) * -10) - .5)
    ABB = (neg * abiom).sum()
    ABA = (pos * abiom).sum()
    BEA = (pos * biom).sum()
    BEB = (neg * biom).sum()
    NC = neg.sum()
    PC = pos.sum()
    #loss = ABA + BEB# - ABB - BEA
    loss = ABA**4/ABB + BEB**3/BEA #- Works the best so far
    #loss = (ABA**2 + BEB+ (NC+PC)**2)/((ABB+BEA**2)*(NC*PC+1)**.5)
    #loss = ABA
    #print(loss.grad,"GRAD")
    #loss = (ABA**3 + BEB+ (NC+PC)**2)/((ABB+BEA)*(NC*PC+1)**.5)
    #loss = (ABA + BEB+ (NC+PC)**2)/((ABB+BEA)*NC*PC)
    #loss = (ABA + BEB)/(ABB+BEA)
    #loss = (ABA + BEB)/(ABB+BEA)
    #loss = (ABA + BEB+ (NC+PC)**2)/((ABB+BEA)*NC*PC) #lr = 1 works but is too heavily drawn towards median
    #loss = (ABA + BEB+ 10000000)/((ABB+BEA)*NC*PC)
    #makeHist(pos,title=b)
    return(loss)

def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,col)
    ch = int(input(title)) #multiboxplot needs this to trigger an error if non int sent
    uch = sorted(list(obs[obs.columns[ch]].unique()))
    return(ch,uch)


def getThresh(cm,df,obs,t2='',maxReps = 10, LR = 1e-10):
    antiS = cm.idxmin(axis=1)
    thresho = []
    for i,b in enumerate(df.columns):
        print(b)
        abn = antiS.iloc[i]
        switch = 0
        badS = ["R0","DAPI","R5","R6","R7"]
        for bs in badS:
            if bs in b:
                switch = 1
        if switch == 1:
            thresho.append(9999)
            continue
        biom = torch.tensor(list(df.loc[:,b]))
        #makeHist(biom,title=b)
        abiom = torch.tensor(list(df.loc[:,antiS.iloc[i]]))

        iThresh = gridSearch(biom,abiom)
        print(float(iThresh),"initial threshold")
        thresh = torch.tensor(iThresh,requires_grad = True)
        optim = torch.optim.SGD([thresh], lr=LR)
        lloss = 999999999#torch.tensor(999999999)
        loss = 0
        reps = 0
        for i in range(NITERS):
            loss = getLoss(biom,abiom,thresh)
            if i % 50 == 20 and False:
                print(float(loss.detach()),float(thresh.detach()))
                #try:
                #print(float(loss.detach()),float(lloss.detach()))
                #print(lloss/loss)
                #print(lloss/loss <  1 + 10 ** -9)
                #except:
                #print(float(loss.detach()),lloss)

            if lloss > loss or i < MIN:
                #print(float((lloss/loss).detach()))
                lloss = loss
                loss.backward()
                optim.step()
                reps = 0
                #print(loss)
            else:
                reps += 1
                loss.backward()
                grad = thresh.grad
                #print(thresh.is_leaf)
                #print(grad,"grad")
                #print(float(thresh.detach()))
                thresh = thresh - grad * LR * 2
                thresh = torch.tensor(thresh, requires_grad = True)
                #print(thresh.is_leaf,"still leaf?")
                #print(float(thresh.detach()))
                #print(loss, grad, reps)
            if reps > maxReps:
                #print(float((lloss/loss).detach()))
                thr = float(thresh.detach())
                thresho.append(thr)
                #print(float(BEB.detach()),"BEB")
                #print(round(float((lloss/loss).detach()),3),"   lloss/loss", lloss>loss,round(float((lloss-loss).detach()),3),float(lloss.detach())>float(loss.detach()))
                #print(float(thresh.detach()),b,i,"done! Max reps\n")
                print(thr,"thresh",i,"iters\n")
                makeHist(biom,title=b,vline=thr,t2=t2)
                scatterplot(df,obs,b,abn,vline = thr,t2=t2)
                break

            #break
    return(thresho)

def gridSearch(biom,abiom):
    threshs = torch.tensor(np.linspace(0,500,50))
    losses = []
    for thresh in threshs:
        loss = getLoss(biom,abiom,thresh)
        losses.append(loss)
    mind = losses.index(min(losses))
    plt.scatter(np.linspace(0,500,50),losses)
    plt.show()
    return(threshs[mind])




def scatterplot(df,obs,biom,abiom,vline=None,t2=''): #biom and abiom are names, not series (as they are in loss fn)
        fig = plt.figure()
        ax = fig.add_subplot()
        colors = allc.colors
        for i,uo in enumerate(sorted(list(obs.loc[:,"slide_scene"].unique()))):
            key = obs.loc[:,"slide_scene"] == uo
            #print(key)
            color = colors[i]
            ax.plot(df.loc[key,biom],df.loc[key,abiom], color=color, marker='x', linestyle='none', markersize=.25)
            if type(vline) != type(None):
                ax.vlines(x=vline,ymin=min(df.loc[key,abiom]),ymax=max(df.loc[key,abiom]),color="b")
        ax.set_xlabel(biom)
        ax.set_ylabel(abiom)
        ax.set_title(t2)
        plt.show()



def makeHist(series,title='',vline=None,t2=''):
    try:
        series = pd.Series(series.detach().numpy())
    except Exception as e:
        print(e)
    nbin = int(series.shape[0]/100)
    mi,ma = series.min(),series.max()
    intensity = []
    count = []

    for i in np.linspace(mi,ma,nbin):
        intensity.append(i)
        key = series <= i
        count.append(key.sum())
        series = series.loc[~key]
        #print(series.shape)
    hist = pd.DataFrame([intensity,count]).transpose()
    hist.columns = ["intensity","count"]
    fig,ax = plt.subplots()
    ax.plot(hist["intensity"],hist["count"])
    if type(vline) != type(None):
        #print("VL",vline)
        ax.vlines(x=vline,ymin=0,ymax=max(count),color="r")
    ax.set_title(title+' - '+t2)
    plt.show()

    return(hist)

def showHeatmap(cdf):
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    plt.show()


def corMat(df):
    cm = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0])
        cm.append(cl)
    return(pd.DataFrame(cm,columns=df.columns,index = df.columns))


def loadTraining(fold,sep = COL):

    DFs = [] #list of dataframes (one per slide)
    threshs = {} #dict of biom_slide : thresh
    zinfo = {} #dict of lists biom_slide : [mean, stdev]
    for file in os.listdir(fold):
        #if 'MIT' not in file:
        #    continue
        if '_df.csv' in file:
            stem = file.split('_df.csv')[0]
            print(stem)
            for f2 in os.listdir(fold):
                if stem in f2 and '_obs.csv' in f2:
                    df0 = pd.read_csv(fold+'/'+file,index_col=0)
                    obs0 = pd.read_csv(fold+'/'+f2,index_col=0)
                    if sep not in obs0.columns:
                        if 'all data' not in obs0.columns:
                            obs['all data'] = '1'
                        ch,uch = obMenu(obs0,"category whose subcategories will be individually annotated (e.g. batch or patient)")
                        sep = obs.columns[ch]

                    for slide in obs0.loc[:,sep].unique():
                        key = obs0.loc[:,sep] == slide
                        df = df0.loc[key,:]
                        DFs.append(df)
                        obs = obs0.loc[key,:]
                        thresh = getManualThresh(df,obs,slide)
                        threshs.update(thresh)
                        zinfo.update(zinf)
        print(threshs,'threshs')
        print(zinfo,'zinfo')
        return(DFs, OBs ,threshs,zinfo)


def getThresh(df,obs,slide):
    thd = {}
    zinf = {}
    for biom in df.columns:
        mean = np.mean(df.loc[:,biom])
        std = np.std(df.loc[:,biom])
        zinf[biom+'_'+slide] = [mean,std]

        bn = biom.split('_')[0]+'_'
        key = obs['Manual Celltype'].astype(str).str.contains(bn)
        if key.sum() == 0:
            thresh = 9
        else:
            print(key,key.sum())
            pser = df.loc[key,biom]
            #thresh = (pser.min()-mean)/std #ZSCORED HERE #don't zscore- the model should learn that the background level is usually around 1000 regardless of distribution
        print(thresh)
        thd[biom+'_'+slide] = thresh
    return(thd,zinf)


if __name__ == "__main__":
    #folder = r'C:\Users\youm\Desktop\src\BR MFC7 GL data pre 230808 pre vietnam'#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    folder = r'C:\Users\youm\Desktop\src\transformernet_training/'
    #stem = ''#'zzz_hta14'#'tiny_pTMA1' #'196_MCF7'#'95_GL'#'95_GL'#'z3_GL631'#'zzz_hta14'#'hta14bx1_ck7'#'recent june 3 23/PIPELINE_hta14_bx1_with_svm'#'PIPELINE_94'#'PIPELINE_hta14_bx1_99'#'93_hta14_no_neigh'#'93_hta14'#87_LC-4'##'89_LC-4_withN'#''96_LC'#cl56_depth_study_H12'#'96_LC'#'97_mtma2'#'93_hta14'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#
    DFs, OBs, threshs, zinfo = loadTraining(folder)
    df,obs,n = Train(df,obs)




    '''
    tT = np.ones((df.shape[0],5))
    print(tT)
    tT = torch.tensor(tT,requires_grad=True)

    optim = torch.optim.SGD([tT], lr=1e-4)
    #for i in tqdm(range(NITERS)):
    for i in range(NITERS):
        loss = getLoss(bT,wT,tT) #,df.columns
        print(loss)
        loss.backward()
        optim.step()

    obs.to_csv("tiny_pTMA1_obs.csv")
    '''