# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:34:40 2023

@author: youm
"""


'''
from cmifA44:
op = ["start over with raw data","log2","scale from -1 to 1", "z-score","elmarScale","trim outliers",
      "make control TMA sample sizes the same","combat",
      "apply TMA combat to other dataset","equalizeBiomLevel","adjust for negative values",
      "save to csv", "pick subset of data", "manually threshold",
      "cluster by obs catagory","Leiden cluster","GMM cluster","K-means","aggregate",
      "manually celltype random training set","auto-cell-type",
      "convert df to fractions in obs categories","convert to superbiom-only df",
      "remove non-primary biomarkers","calculate biomarker expression in region around each cell",
      "count label fractions in neighborhood","calculate entropy in neighborhood",
      "select ROI","remove cells expressing certain biomarker combinations","pick random subset","clag","clauto"]
fn = [revert,log2,scale1,zscore,elmarScale,outliers,equalizeTMA,combat,TMAcombat,equalizeBiomLevel,remNegatives,save,pick,
      manThresh,obCluster,leiden,gmm,kmeans,aggregate,celltype,autotype,countObs,superBiomDF,
      onlyPrimaries,regionAverage,neighborhoodFractions,neighborhoodEntropy,roi,simulateTherapy,subset,clag,clauto]
'''

import os
import numpy as np
import pandas as pd
import math
import scipy
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture as GMM
import scanpy as sc
import anndata
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import allcolors as allc
#import random
#from sklearn.metrics import silhouette_samples, silhouette_score
import orthoType7 as oT
import cmifAnalysis50 as cm
from tqdm import tqdm
import IFvisualization2 as ifv
from scipy.stats import zscore as ZSC

SAVE = 'ask'
SPATH = r'C:\Users\youm\Desktop\src\unsorted figs'
CATN = ''
PXSIZE = .325

def main(df,obs,dfxy):
    global CATN
    obs["all data"] = 'all data'
    dfs = [df,obs,dfxy]
    dfa = []
    try:
        ch,uch = obMenu(obs,'repeat analysis on each unique value in:')
    except:
        ch = list(obs.columns).index('all data')
        uch = ['all data']
    print(uch,'uch')
    ks = input('key string, if any, for categories to consider (skips others- if blank, processes all)')
    CATN = obs.columns[ch]
    print(CATN,uch)
    #obcol = obs.columns[ch]
    if input("repeat last run? (y)") == 'y':
        nn,commands = lastrun(dfs)
    else:
        nn,commands = mainMenu(dfs)
        if input("save commands? (y)")=='y':
            saveCommands(9,commands,9)
    for uc in uch:
        key = obs.iloc[:,ch] == uc
        sdfs = []
        for d in dfs:
            sdfs.append(d.loc[key,:])

        if ks != '':
            if ks not in uc:
                dfa.append(sdfs)
                continue

        sdfs,nn=mainMenu(sdfs,commands,uc)
        dfa.append(sdfs)
    odfs = []
    for i in range(3):
        bi = []
        for d in dfa:
            bi.append(d[i])
        odfs.append(pd.concat(bi,axis=0))
    return(odfs[0],odfs[1],odfs[2])

'''
main functions
'''


def menu(dfs,options,functions,com=[],cat=''):
    print(com,'com into menu')
    print("we need to split processing from visu so\n visu can do-for-each obs category e.g. make pseudoimage of prolif, then celltype, then a barplot of each.")
    if len(com) == 0:
        coms = []
        while True:
            print("\n")
            for i,op in enumerate(options):
                print(i,op)
            try:
                print("send non-int when done (return to previous menu)")
                ch = int(input("number: "))
            except:
                print(coms,"coms out of menu")
                return([],coms)
            nn,com=functions[ch](dfs,com=[])
            coms.append([ch]+com)

    else:
        for subcom in com:
            if type(subcom) == list:
                ch = subcom[0]
                print('running subcommand:',subcom,options[ch], 'on category',cat)
                dfs,nn = functions[ch](dfs,subcom,cat)
        return(dfs,[])




def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,col)
    ch = int(input(title))
    uch = obs[obs.columns[ch]].unique()
    return(ch,uch)




'''
menus
'''
def mainMenu(dfs,com=[],cat=''):
    print('main menu')
    op = ['general data handling','scaling','clustering','batch-correction',
          'celltyping','neighborhood analysis','apply celltype labels to other labes (e.g. celltype Leiden clusters)',
          'visualize']
    fn = [selection,scaling,clustering,batchCorrection,
          celltyping,neighborhoodAnalysis,clauto,
          visu]
    dfs1,coms=menu(dfs,op,fn,com,cat)
    if len(dfs1) > 0:
        dfs = dfs1
    #print(coms,'coms out from mainMenu')
    return(dfs,coms)

def lastrun(dfs,com=[],cat=''):
    f = open('ifp3_lastrun.txt','r')
    coms = f.readlines()[0]
    f.close()
    print("com read in:",coms,type(coms))
    com = ifv.s_to_l(coms)[0]
    print(com,type(com))
    return(dfs,com)


def saveCommands(dfs=9,com=[],cat=''):
    print(com)
    coms = ifv.l_to_s(com)
    coms = coms.replace("][","],[")
    print(coms,"COMS OUT")
    f = open(r"C:\Users\youm\Desktop\src/ifp3_lastrun.txt",'w')
    f.write(coms)
    f.close
    return(dfs,com)

def s_to_l(coms):
    print(coms,"use ifv version instead")
    input('...')


    ocom = []
    print(ocom,"ocom")
    i = 0
    while i < len(coms):
        ch = coms[i]
        if ch == "[":
            inbrkt = ""
            nbrkt = 0
            j = i+1
            while True:
                ch2 = coms[j]

                if nbrkt == 0 and ch2 == "]":
                    break
                if ch2 == "[":
                    nbrkt += 1
                elif ch2 == "]":
                    nbrkt -= 1
                inbrkt += ch2
                j += 1
            ocom.append(s_to_l(inbrkt))
            i = j

        elif ch == "]":
            i += 1
        else:
            cs = ""
            j = i
            while True:
                ch2 = coms[j]
                j += 1
                if ch2 == ",":
                    if len(cs) > 0 and "." in cs:
                        ocom.append(float(cs))
                    elif len(cs)>0:
                        if cs == "False":
                            ocom.append(False)
                        else:
                            try:
                                ocom.append(int(cs))
                            except:
                                ocom.append(str(cs))
                    break
                else:
                    cs+=ch2
            i  = j
    return(ocom)




def l_to_s(com,outs = ""):
    print('use ifv version instead')
    input('...')
    outs += "["
    for item in com:
        #print(outs)
        #print(item,'\n')
        if type(item) == list:
            outs += l_to_s(item)
        else:
            outs += str(item)+','
    outs += "]"
    return(outs)


def checkChange(s,cat):
    if input(cat+":\n"+s+"\nchange? (y):") == 'y':
        return(input(": "))
    else:
        return(s)


def saveF(data,foln,filn,typ="png"):
    badS = [':']
    for bs in badS:
        if bs in filn:
            filn = filn.replace(":",".")
        if bs in foln:
            foln = foln.replace(":",".")
    if not os.path.isdir(SPATH+"/"+foln):
        if not os.path.isdir(SPATH):
            os.mkdir(SPATH)
        os.mkdir(SPATH+"/"+foln)
    if typ == "png":
        return(SPATH+"/"+foln+"/"+filn+'.png')


def selection(dfs,com=[],cat=''):
    print('selection (general data handling)')
    op = ['save to csv','remove nan values and zero columns'] #'save in ram','revert to ram save','pick subset of data'
    fn  = [save,ifv.autoClean,pick]
    dfs,com=menu(dfs,op,fn,com,cat)
    #print(com,'com in selection')
    return(dfs,com)



def scaling(dfs,com,cat=''):
    print("scaling")
    op = ['zscore across samples (cells)']
    fn=[zscore]
    dfs,com=menu(dfs,op,fn,com,cat)
    return(dfs,com)

def clustering(dfs,com,cat=''):
    op = ["K-Means","Leiden","GMM"]
    fn = [kmeans,autoleiden,gmm]
    dfs,com = menu(dfs,op,fn,com,cat)
    return(dfs,com)

def batchCorrection():
    pass

def celltyping(dfs,com=[],cat='',clean=True):
    print('celltyping')
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs,['n'])
        print(dfs[0].shape,dfs[0].columns)

    op = ['SD-type','orthogonality type','add labels to existing biomarker phenotype','Maxey type']
    fn = [autotype,orthoType,labelPhenotype,maxeyType]
    dfs,com=menu(dfs,op,fn,com,cat)
    #print(com,'com out from mainMenu')
    return(dfs,com)

def neighborhoodAnalysis(dfs,com=[],cat=''):
    print('neighborhood analysis')
    op = ['calculate protein expression in radius','count celltypes in radius']
    fn = [regionAverage,neighborhoodCount]
    dfs,com=menu(dfs,op,fn,com,cat)
    return(dfs,com)

def visu(dfs,com=[],cat=''):
    global SAVE
    global SPATH
    if len(com) == 0:
        spath = input('save location:')

        catlist = ifv.getCats(dfs[1],required=False,title='Columns to color figures by (or sort x axis for boxplot). Send none to load last set.')
        nn,commands = ifv.mainMenu(dfs)

        return([],[catlist,commands,spath])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    catlist,commands,spath = com[1],com[2],com[3]
    spath = spath+'/'+cat
    print('save path!!!',spath)
    df,obs,dfxy = ifv.main(df,obs,dfxy,spath=spath,catlist=catlist,commands=commands)
    return([df,obs,dfxy],com)





'''
neighborhood analysis
'''
def regionAverage(dfs,com=[],cat=''): #make only check same slidescene
    if len(com) == 0:
        while True:
            try:
                radius = float(input("radius (um) to consider in average: "))/PXSIZE
                return([],[radius])
            except:
                print("invalid radius, send number")
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    radius = com[1]
    ndf = []
    for us in obs["slide_scene"].unique():
        key0 = obs["slide_scene"] == us
        tdfxy = dfxy.loc[key0,:]
        tdf = df.loc[key0,:]
        for i in range(tdfxy.shape[0]):
            if i % 1000 == 500:
                print(i/tdfxy.shape[0]*100,"% done with",us)
            neighbors = []
            x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
            nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
            distanceV = ((x-nx)**2+(y-ny)**2)**.5
            key = distanceV < radius
            neighbors = tdf.loc[key,:]
            neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
            if neighbors.shape[0]> 1:
                avg = neighbors.mean(axis=0)
                ndf.append(pd.DataFrame(avg.values,index=df.columns,columns = [pd.Series(tdf.index).iloc[i]]).transpose())
            else:
                ndf.append(pd.DataFrame(columns =df.columns ,index=[pd.Series(tdf.index).iloc[i]] ))#
    ndf = pd.concat(ndf,axis=0)
    print("ndf start",ndf,"ndf end")
    for biom in ndf.columns:
        print(biom,"biom")
        ndf.loc[:,biom]=ndf.loc[:,biom].fillna(0)
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius*PXSIZE),axis=1)
    df= pd.concat([df,ndf],axis=1)
    return([df,obs,dfxy],[])




def neighborhoodCount():
    pass

'''
scaling
'''

def zscore(dfs,com=[],cat=''):
    if len(com) == 0:
        return([],[])
    dfs[0] = scipy.stats.zscore(dfs[0])
    return(dfs,[])


'''
selection and annotation
'''


def save(dfs,com=[],cat=''):
    print('saving')
    if len(com) == 0:
        filename = input("filename?")
        return([],[filename])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    filename = com[1]+'_'+cat
    print(filename)
    df.to_csv(filename+"_df.csv")
    obs.to_csv(filename+"_obs.csv")
    dfxy.to_csv(filename+"_dfxy.csv")
    return(dfs,[])

def pick(dfs,com=[],cat=''):
    print('picking subset')



def clauto(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        ch,uch=obMenu(obs,"obs category to auto-annotate cell types")
        res = float(input('resolution:'))
        return([],[ch,res])
    ch,res = com[1],com[2]
    obs = obs.astype(str)
    print(obs.shape)
    oobs = obs.copy()
    chs, uchs = [ch],[obs.iloc[:,ch].unique()]
    for i,ch in enumerate(chs):
        uch = uchs[i]
        adf,aobs,axy = clag(df,obs,dfxy,ch,uch)
        dfs,xx = autotype([adf,aobs,axy],['nn',res,False],cat=cat,name=obs.columns[ch]+"cluster autotype",chanT=False)
        aobs = dfs[1]
        #x,aobs,xx = autotype(adf,aobs,axy,chanT=False,name=obs.columns[ch]+"cluster autotype",res=res)
        print(obs.shape)
        for col in aobs.columns:
            if obs.columns[ch]+"cluster autotype" in col:
                print(col,aobs.loc[:,col].unique(),"!!")
                obs[col] = ""
                for uc in aobs.index:
                    key = obs.iloc[:,ch] == uc
                    obs.loc[key,col] = aobs.loc[uc,col]
        print(obs.shape,df.shape)
        #print(obs,df)
    return([df,obs,dfxy],[])

def clag(df,obs,dfxy,ch=None,uch=None,z=True):
    if not ch:
        ch,uch=obMenu(obs,"obs category to auto-annotate cell types")

    if z:
        zdf,zobs,zxy =zscore1(df,obs,dfxy,ax=0)
    else:
        zdf,zobs,zxy = df,obs,dfxy
    ocol = obs.columns[ch]
    ndf,nobs,nxy = [],[],[]
    for uc in uch:
        key = zobs.loc[:,ocol] == uc
        sdf = zdf.loc[key,:]
        sobs = zobs.loc[key,:]
        sxy = zxy.loc[key,:]
        ndf.append(sdf.mean(axis=0))
        nxy.append(sxy.mean(axis=0))
        #print(sobs.mode(axis=0).iloc[0,:],"/n/n")
        #time.sleep(1)
        nobs.append(sobs.mode(axis=0).iloc[0,:])
    dfs = [ndf,nobs,nxy]
    for i,d in enumerate(dfs):
        dfs[i] =pd.concat(d,axis=1).transpose()
        dfs[i].index = uch.astype(str)
        #print(dfs[i].columns)
        #print(dfs[i].shape)
        #print(dfs[i])
    return(dfs[0],dfs[1],dfs[2])


def maxeyType(dfs,com=[],cat='',
              method = 'zscore',
              fileName = 'primary_celltype.csv',typeName = 'Primary Celltype: Matrix', default = '5: stromal'):
    if len(com) == 0:
        return([],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    fold = r'C:\Users\youm\Desktop\src\maxey matrices'
    prim = pd.read_csv(fold+'/'+fileName,index_col=0)
    prim = prim.fillna(0)
    #markers = [mark.split('_')[0] for mark in df.columns]
    dm = {}
    for mark in prim.columns:#markers:
        for col in df.columns:
            if mark+'_' in col:
                dm[mark] = df.loc[:,col]
                break
    #print(dm)
    #print(dm.values())
    #print('\n\n', pd.concat(dm.values(),axis=1),'\n\n')
    dm1 = pd.concat(dm.values(),axis=1)
    dm1.columns = dm.keys()
    dm = dm1
    #print(dm)
    prim = prim.loc[:,dm.columns]
    print(prim)
    psum = prim.sum(axis=1)
    print(prim.sum(axis=1))
    prim = prim.loc[psum > 0,:]
    psum = prim.sum(axis=1)
    prim = prim.divide(psum,axis=0)
    print(prim)
    #input()
    obs[typeName] = default

    if method == 'rank':
        threshold = .7                                      #!!!!!!!!!!!!!!!!!
        for col in dm.columns:
            scores = np.arange(dm.shape[0])/dm.shape[0]
            inds = dm.loc[:,col].sort_values()
            dm.loc[inds.index,col] = scores
    elif method == 'zscore':
        threshold = .01                                      #!!!!!!!!!!!!!!!!!!!
        dm = dm.apply(ZSC)
        dm = dm.clip(lower=0)
    print(dm)


    types = list(prim.index)
    print(types)

    for i in tqdm(range(df.shape[0]),'assigning celltypes from: '+fileName): #df index should match dm index

        cell = dm.iloc[i,:]
        #print(cell.shape,prim.shape)
        scores = np.matmul(cell.values,prim.T.values)
        #print(cell)
        print(scores)
        if np.amax(scores) < threshold:
            continue
        obs.loc[dm.index[i],typeName] = types[np.argmax(scores)]
        print(types[np.argmax(scores)])

        #input()
    return([df,obs,dfxy],[])



def maxeyTypeNotes(dfs,com=[],cat=''):
    #Explained by Jessica Maxey from 1/13/25
    print('for each px/cell, calculate fraction of px/cells below the subject"s expression level')
    print('give -.1 for cells with expression below threshold (e.g. 50th percentile and below goes from 0.499 -> -0.1')
    print('make binary matrix of which markers are used as primary markers for which celltype')
    print('multiply score matrix by binary matrix to get score for each celltype. tiebreak with hierarchy.')

def labelPhenotype(dfs,com=[],cat=''):
    if len(com) == 0:
        ch,uch = obMenu(dfs[1],"column with phenotype labels")
        return([],[dfs[1].columns[ch]])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    ncn = com[1]
    obs = parseTypes(df,obs,dfxy,column = ncn)
    obs = parseSecondary(df,obs,dfxy,column = ncn)
    return([df,obs,dfxy],[])


def orthoType(dfs,com=[],cat=''):
    if len(com) == 0:
        return([],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    df,obs,dfxy = oT.main(df,obs,dfxy,sep=CATN)
    ncn = "orthoThresh phenotype " + CATN
    obs = parseTypes(df,obs,dfxy,column = ncn)
    obs = parseSecondary(df,obs,dfxy,column = ncn)
    return([df,obs,dfxy],[])

def autotype(dfs,com,cat='',chanT=True,name="autoCellType res: ",res=None): #the old version that keeps more information is in cmifAnalysis36
    if len(com) == 0:
        inp = input("compare to channel threshold? (y)")
        if inp == "y":
            chanT = True
        else:
            chanT = False
        if not res:
            try:
                res= float(input("number of standard deviations above mean required to count as +,(send non-int to enter custom res for each celltype): "))
            except:
                typs = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","secondary markers (e.g. ki67)"]
                ress = []
                for t in typs:
                    print(t)
                    res= float(input("number of standard deviations above mean required to count as + for "+t+" :"))
                    ress.append(res)
                return([],[ress,chanT])
        return([],[res,chanT])
    print(com)
    res = com[1]
    chanT = bool(com[2])
    if chanT == "False":
        chanT = False
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    roundThresh = [1500,1250,1000,750]
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    nbr = [['H3K27', 'PCNA', 'R6Qc2', 'LamB1', 'R0c2', 'pHH3', 'FN', 'R1Qc2', 'GFAP', 'Myelin', 'S100A', 'CAV1', 'Glut1', 'Vim', 'NeuN', 'aSMA', 'panCK'],
        ['H3K4', 'TUBB3', 'R6Qc3', 'CD68', 'R0c3', 'pMYC', 'CD11b', 'R1Qc3', 'CTNNB', 'PDL1', 'CD56', 'CD11c', 'CD133', 'HLA-DR', 'CD90', 'CD8', 'p53'],
        ['ColIV', 'CD163', 'R6Qc4', 'CD45', 'R0c4', 'Ki67', 'gH2AX', 'R1Qc4', 'IBA1', 'p63', 'BCL2', 'PD1', 'pMYCab', 'MSH6', 'CD31', 'CD4', 'pRPA'],
        ['ColI', 'BMP2', 'R6Qc5', 'CD20', 'R0c5', 'CD3', 'CD44', 'R1Qc5', 'LamAC', 'GRNZB', 'Rad51', 'CGA', 'CSF1R', '53BP1', 'YAP1', 'FoxP3', 'ZEB1']]
    for i in range(len(nbr)):
        biomRounds[i] += nbr[i]
    odf = df.copy()

    #chanT = False
    if chanT:
        #key2 = pd.DataFrame(data=np.zeros_like(odf),columns=odf.columns,index=odf.index)
        key2 = pd.DataFrame(data=np.ones_like(df),columns=df.columns,index=df.index)
        #means = df.mean(axis=0)
        #sds = df.std(axis=0)
        #zSer = pd.Series(index = df.columns,data=means+sds*res)
        #print(zSer)
        for i,roun in enumerate(biomRounds):
            rawThresh = roundThresh[i]
            for bIm in roun:
                for bim in df.columns:
                    if bIm+"_" in bim:
                        #key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
                        key2.loc[df.loc[:,bim]<=rawThresh,bim] = 0
                        #bimT = zSer.loc[bim]
                        #key2.loc[odf.loc[:,bim]>bimT,bim] = 1
                        #print("ding")
    if type(res) == list:
        resN = "multi"
    else:
        resN = str(res)
    #print(list(key2.iloc[:,0]),"ar key")
    obs[name+resN] = " "
    df,obs,dfxy = zscore1(df,obs,dfxy,ax=0)
    #print(df)
    mapp = {}
    toThresh = []
    for biom in df.columns:
        if "neigh" in biom:
            continue
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    toThresh = list(mapp.keys())
    #others = ["Ki67", "PCNA", "pHH3","pRB","ER","PgR","AR","HER2","Fox","GRNZB","aSMA","Vim","VIM","ColI","PD1"] #CAV
    others = list(df.columns)
    for biom in df.columns:
        for o in others:
            if o in biom:
                toThresh.append(biom)
    for biom in tqdm(toThresh,'extracting markers'):
        obs[biom+'+'] = 'no'
        if type(res) == list:
            typs = ["1 endothelial","2 immune","3 tumor","4 active fibroblast"]
            btype = fillMap(biom)
            if btype in typs:
                res1 = res[typs.index(btype)]
            else:
                res1 = res[-1]
        else:
            res1 = res
        if chanT:
            key3 = key2[biom] == 1
            key = df[biom]>res1
            print(biom,any(key),any(key3),any(key & key3))
            obs.loc[key & key3,name+resN] += biom + " "
            obs.loc[key,biom+'+'] = 'yes'
        else:
            key = df[biom]>res1
            obs.loc[key,name+resN] += biom + " "
            obs.loc[key,biom+'+'] = 'yes'

    obs = parseTypes(df,obs,dfxy,column=name+resN)
    print(obs[name+resN].unique(),"uobs")
    obs = parseSecondary(df,obs,dfxy,column=name+resN)
    #if input("keep z-scoring?") == "y":
        #return(df,obs,dfxy)
    dfs = [odf,obs,dfxy]
    return(dfs,[])


def fillMap(biom,TONLY=False):
    bTypes = [["1 endothelial",["CD31","CAV1"]],
              ["2 immune",["CD11_","CD20","CD3","CD4_","CD45","CD6","CD8","F480"]],
              ["3 tumor",["CK","Ecad","MUC1","HER","TUBB","CD113",'GFAP','CTNNB','NeuN','YAP1', 'Myelin','Amy','EGFR']],    #
              ["4 active fibroblast",["aSMA","Vim","VIM","ColI_","CD90"]]]
    if TONLY == True:
        print("FILLMAP IS FOR TUMOR ONLY RN")
        bTypes = [["3 tumor",["CK","Ecad","MUC1",'EGFR',"HER","TUBB","CD113",'GFAP','CTNNB','NeuN','YAP1', 'Myelin']],
                  ]
    for typeA in bTypes:
        for stem in typeA[-1]:
            if "CD44" in biom or "in radius" in biom or "neighbors" in biom:
                return(None)
            if stem in biom:
                return(typeA[0])

def parseTypes(df,obs,dfxy,column="none",TONLY=False):
    #if column == 'none':
        #ch,uch = obMenu("column to apply types to")
        #column = obs.columns[ch]
    mapp = {}
    for biom in df.columns:
        cType = fillMap(biom,TONLY=TONLY)
        if cType != None:
            mapp[biom]=cType
    #print(mapp)
    mapp = {k: v for k, v in sorted(mapp.items(), key=lambda item: item[1])}
    #print(mapp)
    if TONLY:
        obs["Primary Celltype "+column] = "3 tumor"
    else:
        obs["Primary Celltype "+column] = "5 stromal"
    for biom in mapp.keys():
        keyCol = obs[column].str.contains(biom)
        #print(list(keyCol))
        unasKey = obs["Primary Celltype "+column] == "5 stromal"
        obs.loc[keyCol & unasKey,"Primary Celltype "+column] = mapp[biom]
    return(obs)

def parseSecondary(df,obs,dfxy,column):
    #column must be column with phenotypes
    #must have "Primary Celltype " + column labels

    uch = obs[column].unique()
    #print(uch,"uch")
    #obs["Secondary Celltype"+column] = obs["Primary Celltype "+column] #cmif39 has this version
    obs["proliferating "+column] = "no"
    obs["tumor subtype "+column] = np.nan
    obs["receptors "+column] = np.nan
    obs["immune subtype "+column] = np.nan
    obs["immune checkpoints "+column] = np.nan
    obs["cytotoxic "+column] = np.nan
    #obs["fibroblast type "+column] = np.nan



    proL = ["Ki67","PCNA","pHH3","pRB",'pMYC',"Ki-67"]#
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3_ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68","CD163",'F480'] #ADD CSF1R?
    Hl = ['ER_', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox","FOX","PD-1"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_","SMA"]
    '''
    proL = ["Ki67","pHH3","PCNA"]#"pRB"
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3+ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68","IBA1"] #ADD CSF1R?
    Hl = ['ER', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_"]
    '''
    #checkL changed so first char must be the same!!! HER2 vs ER
    for typ in tqdm(uch,'extracting secondary celltypes'):
        key = obs[column] == typ
        typeD = {'pro':0,'Lum':0,'Bas':0,"HR":0,"HER":0,'T-c4':0,'T-c8':0,'T-c3':0,
                 'B-c':0,'Mac':0,"CheckP":0,"CytoT":0,"activeFB":0,"Mesen":0}
        if checkL(typ,proL):
            typeD['pro'] = 1
        if checkL(typ,mesL):
            typeD["Mesen"] = 1
        if typeD["Mesen"] == 0:
            if checkL(typ,lumL):
                typeD["Lum"] = 1
            if checkL(typ,basL):
                typeD["Bas"] = 1
        if checkL(typ,Hl):
            typeD['HR'] = 1
        if checkL(typ,HEl):
            typeD['HER'] = 1
        if checkL(typ,cpL):
            typeD["CheckP"] = 1
        if checkL(typ,cytL):
            typeD["CytoT"] = 1
        if checkL(typ,TL4):
            typeD['T-c4'] = 1
        elif checkL(typ,TL8):
            typeD['T-c8'] = 1
        elif checkL(typ,TL3):
            typeD['T-c3'] = 1
        elif checkL(typ,BL):
            typeD['B-c'] = 1
        elif checkL(typ,macL):
            typeD['Mac'] = 1
        #if checkL(typ,acL):
            #typeD["activeFB"] = 1
        #print(typ,typeD)
        #print(typ,typeD)
        recSwitch = 0
        tuseSwitch = 0
        for sty in typeD.keys():
            if typeD[sty] == 1:
                if sty in "Mesen":
                    pKey = obs["Primary Celltype "+column] == "3 tumor"
                    obs.loc[key & pKey,"tumor subtype "+column] = sty
                if sty in "Lum Bas":
                    #print('ding')
                    pKey = obs["Primary Celltype "+column] == "3 tumor"
                    if tuseSwitch == 0:
                        obs.loc[key & pKey,"tumor subtype "+column] = sty + " "
                        #print(obs.loc[key & pKey,"tumor subtype "+column])
                        tuseSwitch = 1
                    else:
                        obs.loc[key & pKey,"tumor subtype "+column] = obs.loc[key & pKey,"tumor subtype "+column]+ sty + " "
                        #print(obs.loc[key & pKey,"tumor subtype "+column])
                if sty in "HR HER":
                    pKey = obs["Primary Celltype "+column] == "3 tumor"
                    if recSwitch != 0:
                        obs.loc[key & pKey,"receptors "+column] =obs.loc[key & pKey,"receptors "+column]+" "+ sty
                        #print(any(key&pKey))
                    else:
                        obs.loc[key & pKey,"receptors "+column] = sty
                        recSwitch = 1
                        #print(any(key&pKey),"!")
                if sty in "pro":
                    obs.loc[key,"proliferating "+column] = "yes"
                if sty in "CheckP":
                    pKey = obs["Primary Celltype "+column] == "2 immune"
                    obs.loc[key & pKey,"immune checkpoints "+column] = "yes"
                if sty in "CytoT":
                    pKey = obs["Primary Celltype "+column] == "2 immune"
                    obs.loc[key & pKey,"cytotoxic "+column] = "yes"
                if sty in "T-c4 T-c8 T-c3 B-c Mac":
                    pKey = obs["Primary Celltype "+column] == "2 immune"
                    obs.loc[key & pKey,"immune subtype "+column] = sty
                #if sty in "activeFB":
                    #pKey = obs["Primary Celltype "+column] == "4 stromal"
                    #obs.loc[key & pKey,"fibroblast type "+column] = "active FB"
                #else:
                    #obs.loc[key,"Secondary Celltype"+column] +=" "+ sty
    pKey = obs["Primary Celltype "+column] == "3 tumor"
    key =pd.isna( obs["tumor subtype " + column])
    #print(key.sum(),"number of np.nan")
    obs.loc[key & pKey,"tumor subtype " + column] = "Negative"

    #print(obs.loc[:,"receptors " + column])
    pKey = obs["Primary Celltype "+column] == "3 tumor"
    key = pd.isna(obs["receptors " + column])
    obs.loc[key & pKey,"receptors " + column] = "TN"
    #print(obs.loc[:,"receptors " + column])

    pKey = obs["Primary Celltype "+column] == "2 immune"
    key = pd.isna(obs["cytotoxic " + column])
    obs.loc[key & pKey,"cytotoxic " + column] = "no"

    pKey = obs["Primary Celltype "+column] == "2 immune"
    key = pd.isna(obs["immune subtype " + column])
    obs.loc[key & pKey,"immune subtype " + column] = "Unclassified immune"

    pKey = obs["Primary Celltype "+column] == "2 immune"
    key = pd.isna(obs["immune checkpoints " + column])
    obs.loc[key & pKey,"immune checkpoints " + column] = "no"

    #pKey = obs["Primary Celltype "+column] == "4 stromal"
    #key = pd.isna(obs["fibroblast type "+column])
    #obs.loc[key & pKey,"fibroblast type " + column] = "support FB"
    #print(list(obs.loc[:,"tumor subtype "+column])[:50],column)
    return(obs)

def checkL(biomsS,lis):
    for ent in lis:
        if ent in biomsS: #and ent[0] == biomsS[0]: #biomS has all positives - long string
            #print("donmg")
            return(True)
    return(False)


def zscore1(df,obs,dfxy,ax=None):
    vals = df.values
    shape = vals.shape
    if ax == None:
        print("0 for vertical (by protein), 1 for horizontal")
        ax = int(input("axis (0/1):"))
    newA = np.zeros(shape)
    if ax == 0:
        for i in range(shape[1]):
            col = vals[:,i].tolist()
            try:
                zCol = zScoreL(col)
            except:
                zCol = list(np.zeros(len(col)))
            newA[:,i] = zCol
    if ax == 1:
        for i in range(shape[0]):
            col = vals[i,:].tolist()
            zCol = zScoreL(col)
            newA[i,:] = zCol
    return(pd.DataFrame(data=newA,columns=df.columns,index=df.index),obs,dfxy)

def zScoreL(lis):
    newLis = []
    mean = stat.mean(lis)
    std = stat.stdev(lis)
    if std == 0:
        return(np.zeros(len(lis)))
    for i in lis:
        newLis.append((i-mean)/std)
    return(newLis)

'''
clustering
'''

def kmeans(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        return([],[ncl])
    df,obs = dfs[0],dfs[1]
    ncl = com[1]
    km = KMeans(n_clusters=ncl)
    km.fit(df)
    obs["Kmeans "+str(ncl)] = km.labels_
    return([dfs[0],obs,dfs[2]],[])


def leiden(dfs,com=[],cat=''):
    if len(com) == 0:
        res = float(input("recluster with resolution:"))
        return([],[res])
    res = com[1]
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    print(all(obs.index==df.index),"all index the same")
    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata, key_added='Cluster', resolution=res)
    cn = "Leiden_"+str(res)
    obs[cn] = adata.obs["Cluster"]
    obs[cn] = obs[cn].astype(str)
    return([df,obs,dfxy],[])



def autoleiden(dfs,com=[],cat=''):
    if len(com) == 0:
        target = int(input("get n clusters:"))
        res = float(input("starting Leiden resolution:"))

        return([],[res,target])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    res,target = com[1],com[2]
    incr = res/4
    ncl = 99
    tes = []
    target = int(target)
    ret = 0
    rthre = 10
    while ncl != target:
        if ret > 50:
            target -= 1
            ret = 0
        ret += 1
        print(res,incr)

        #print("!! running with res",res)
        adata = anndata.AnnData(df,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        obs.loc[:,"Leiden_n" + str(target)] = ""
        obs.loc[:,"Leiden_n" + str(target)] = adata.obs["Cluster"].astype(str)
        ncl = len(list(obs.loc[:,"Leiden_n" + str(target)].unique()))
        if ret > rthre:
            rthre += 10
            print("got",ncl,"clusters!")
        tes.append(res)
        if incr < 10**-15:
            incr *= 5
        if ncl > target:
            res -= incr
            if res in tes:
                incr = incr/2
                res += incr
        if ncl < target:
            res += incr
            if res in tes:
                incr = incr/2
                res -= incr
    return([df,obs,dfxy],[])




def gmm(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        return([],[ncl])
    ncl = com[1]
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    #ctypes = ['full','tied','diag','spherical']
    gmm = GMM(n_components=ncl).fit(df)
    obs["GMM_"+str(ncl)] = gmm.predict(df)
    return([df,obs,dfxy],[])



if __name__ == "__main__":
    #'''#"zzz_hta14_tumorneighborhoodcts1"
    folder = r"C:\Users\youm\Desktop\src"   #BR MFC7 GL data pre 230808 pre vietnam"#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    stem = 'm4_cntl_8'#'iy_hta14'#'196_MCF7'#'PIPELINE_hta14_bx1_99'#'93_hta14'#'89_LC-4_withN'#'cl56_depth_study_H12'#''96_LC'#'96_LC'#'97_mtma2'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#

    stem = folder+"/"+stem
    print(stem)
    df = pd.read_csv(stem+"_df.csv",index_col=0)
    obs = pd.read_csv(stem+"_obs.csv",index_col=0).astype(str)
    dfxy = pd.read_csv(stem+"_dfxy.csv",index_col=0)
    print(df.shape[0],"cells")
    while True:
        df,obs,dfxy = main(df,obs,dfxy)
        if input("continue to old analysis tool? (y)") == "y":
            break
    cm.main(df,obs,dfxy)