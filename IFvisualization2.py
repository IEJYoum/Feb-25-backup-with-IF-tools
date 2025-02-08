# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:34:40 2023

@author: youm
"""

import numpy as np
import pandas as pd
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import scanpy as sc
import anndata
import math
import seaborn as sns
#import phenograph  #problem with igraph again
#from scipy import sparse
#from sklearn.metrics import adjusted_rand_score
#import sklearn as skl
#import re
import scipy as sp
import statistics as stat
import random
#import bokehClusterMap1 as bcm
import allcolors as allc
#import sys
#import orthogonal7 as ort
#import combat1 as combat1
#import recropTma as rec
#import lithresh1 as lithresh
#import RESTORE as RES
#import IFanalysisPackage0 as IF
#from skimage import io
#import napari7 as NP
#import napari as NAPARI
#from sklearn.mixture import BayesianGaussianMixture as GMM
import matplotlib.style
import matplotlib as mpl
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
import skimage
#import PIL
#import tifffile
from tqdm import tqdm
#import orthoType5 as ortho
import scipy
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import zscore as ZSC
import cmifAnalysis50 as cm



SPATH = r'W:\ChinData\Cyclic_Analysis\WOO\Figures IY\all_2023\Figures'
#'T:\Cyclic_Analysis\cmIF_2023-04-07_pTMA/new figs IY\pTMA-2'
#r'T:\Cyclic_Analysis\cmIF_2024-10-15_PT5313\figs IY'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY\MCF7 2024-10 all cells Global Thresh'#r'T:\visium\Isaac\New figs'#r'T:\visium\Isaac\JK-2024-04-03'
#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC\figures IY\NEIGHBORHOOD ANALYSIS'#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC\figures IY'
#r'T:\Cyclic_Analysis\KLF4_Project\figures IY\March 24 final/all cells'#r'C:\Users\youm\Desktop\src/ifv output'#r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Analysis\KLF4_Project\figures IY\MSnT'
TSTEM = 'test_251401_300000'#'z06_W'#'march_KLF_08'#'temp'
MAXcATS = 50
SAVE = True
DEVMODE = False
DONE = []
CATS = []

def main(df,obs,dfxy,spath=None,catlist=None,commands=None,clean=True):
    global SPATH
    global UMAP
    global CATS
    global DONE
    print('visu')
    DONE.clear()
    CATS.clear()

    obs = obs.astype(str)
    dfs = [df,obs,dfxy]
    if clean:
        dfs,n = autoClean(dfs)

    if spath:
        SPATH = spath
    print(SPATH,'saving here SPATH!!!')
    if not DEVMODE and not spath:
        SPATH = checkChange(SPATH,'save folder')

    if not catlist:
        catlist = getCats(dfs[1],required=False,title='Columns to color figures by (or sort x axis for boxplot). Send none to load last set.')
        if len(catlist) == 0:
            nn,catlist = lastrun(title='color_categories')
            print(catlist,'CATS')
        else:
            saveCommands(com=catlist,title='color_categories')
    #make this autosave and autoload if user hits enter
    ncl = []
    for cat in catlist:
        if len(dfs[1].loc[:,cat].unique()) <= MAXcATS:
            ncl.append(cat)
    catlist = ncl
    CATS = catlist

    if not commands:
        if input("repeat last run? (y)") == 'y':
            nn,commands = lastrun(dfs)
        else:
            nn,commands = mainMenu(dfs)
            if input("save commands? (y)")=='y':
                saveCommands(9,commands,9)
    for cat in catlist:
        #dfs,nn=mainMenu(dfs,commands,cat)
        mainMenu(dfs,commands,cat)
    return(df,obs,dfxy)

def autoClean(DFs,com=['n'],cat=''): #duplicated in IFA4, IFV2 except it takes df,obs,dfxy
    df,obs,dfxy = DFs[0],DFs[1],DFs[2]
    if len(com) == 0:
        return([],[])
    cho = 1 #drop 0:cells   1:columns(bioms)
    ch = 90 #"max missing % threshold integer (0 to drop all cells with missing values, 100 to keep all
    while ch > 0:
        if cho == 0:
            counts = df.isnull().sum(axis=1)
            #print(counts,counts.shape,df.shape)
            Mx = df.shape[1]
            pts = (np.ones(counts.shape[0]) - counts/Mx)*100
            pts = pd.Series(pts)
            key = pts >= 100-ch
            df = df.loc[key,:]
            obs = obs.loc[key,:]
            dfxy = dfxy.loc[key,:]
            cho = 1
            ch -= 10
        else:
            counts = df.isnull().sum(axis=0)
            #print(counts,counts.shape,df.shape)
            Mx = df.shape[0]
            pts = (np.ones(counts.shape[0]) - counts/Mx)*100
            pts = pd.Series(pts)
            key = pts >= 100-ch
            df = df.loc[:,key]
            cho = 0
            ch -= 10
    #for co in df.columns:
    #    print(co,df[co].isna().sum()/df.shape[0]*100)
    #print(counts,counts.shape,df.shape)
    for col in df.columns:
        if df.loc[:,col].sum() == 0:
            df = df.drop([col],axis=1)
            print('dropping',col)
    return([df,obs,dfxy],[])

def mainMenu(dfs,com=[],cat=''):
    print('main menu')
    op = ['spatial pseudoimage','U-map','barplot','[] boxplot','error bar plot','histogram','cluster heatmap',
          'biomarker sorted heatmap', 'annotation sorted heatmap']
    fn = [spatialLite,showUmap,barplot,boxplot,errorBar,hist,heatmap,biomSortedMap,sortedMap]
    dfs1,coms=menu(dfs,op,fn,com,cat)
    if len(dfs1) > 0:
        dfs = dfs1
    #print(coms,'coms out from mainMenu')
    return(dfs,coms)


def menu(dfs,options,functions,com=[],cat=''):
    print(com,'com into menu')
    #print("we need to split processing from visu so\n visu can do-for-each obs category e.g. make pseudoimage of prolif, then celltype, then a barplot of each.")
    if len(com) == 0:
        coms = []
        while True:
            print("\n")
            for i,op in enumerate(options):
                print(i,":",op)
            try:
                print("send non-int when done (return to previous menu)")
                ch = int(input("number: "))
            except:
                print(coms,"coms out of menu")
                return([],coms)
            nn,com=functions[ch](dfs,com=[])
            coms.append([ch]+com)

    else:
        print(com,'executing com')
        for subcom in com:
            print(com)
            if type(subcom) == list:
                ch = subcom[0]
                print('running subcommand:',subcom,options[ch], 'on category',cat)
                #dfs,nn = functions[ch](dfs,subcom,cat)
                mpl.style.use('default')
                functions[ch](dfs,subcom,cat)
        return(dfs,[])

def checkChange(s,cat='string',paths=False):
    if input(cat+":\n"+s+"\nchange? (y):") == 'y':
        return(input(": "))
    else:
        return(s)


def lastrun(dfs=9,com=[],cat='',title='lastrun'):
    f = open("ifv2_"+title+".txt",'r')
    coms = f.readlines()[0]
    f.close()
    print("com read in:",coms,type(coms))
    com = s_to_l(coms)[0]
    print(com,type(com))
    return(dfs,com)


def saveCommands(dfs=9,com=[],cat='',title='lastrun'):
    print(com)
    coms = l_to_s(com)
    coms = coms.replace("][","],[")
    print(coms,"COMS OUT")
    f = open("ifv2_"+title+".txt",'w')
    f.write(coms)
    f.close
    return(dfs,com)

def s_to_l(coms):
    #print(coms,"s2l")


    ocom = []
    #print(ocom,"ocom")
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
                    switch = 0
                    if len(cs) > 0 and "." in cs:
                        try:
                            ocom.append(float(cs))
                            switch = 1
                        except:
                            pass
                    if len(cs)>0 and switch == 0:
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

    outs += "["
    for item in com:
        #print(outs)
        #print(item,'\n')
        if type(item) == list:
            outs += l_to_s(item)
        elif len(str(item)) == 0:
            outs+= "'"+"'"+"," #this does not seem to work- string gets saved blank.... wait maybe ju
        else:
            outs += str(item)+','
    outs += "]"
    return(outs)


def saveF(data,foln,filn,typ="png"):
    if len(filn) > 105:
        filn = filn[:100]+'_etc_'
    print(foln,filn)
    badS = [':','?','*','<','>',':','|','\ '[0],'/']
    for bs in badS:
        if bs in filn:
            filn = filn.replace(bs,".")
    badS = badS[:-2]
    for bs in badS:
        if bs in foln:
            foln = foln.replace(bs,".")
    if not os.path.isdir(SPATH+"/"+foln):
        #if not os.path.isdir(SPATH):
        #    os.mkdir(SPATH)
        os.makedirs(SPATH+"/"+foln)
    if typ == "png":
        return(SPATH+"/"+foln+"/"+filn+'.png')



def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,col)
    ch = int(input(title)) #multiboxplot needs this to trigger an error if non int sent
    uch = sorted(list(obs[obs.columns[ch]].unique()))
    return(ch,uch)


def getCats(obs,title='',required = True, typ = 'abc'):
    cols = []
    for i,ob in enumerate(obs.columns):
        print(i,ob)
    print('\n'+title+'\n')
    while True:
        imp = input("int or range() to add: ")
        try:
            imp = eval(imp)
            try:
                for im in imp:
                    cols.append(obs.columns[im])
            except:
                cols.append(obs.columns[imp])
        except Exception as e:
            print(cols)
            if required:
                if len(cols) > 0:
                    break
            else:
                i2 = input(str(e)+"\ndone? (y/''):")
                if i2=="y" or i2 == '':
                    break
    nc = []
    print(type(typ))
    if type(typ) == int:
        print('returning col inds')
        for col in cols:
            nc.append(list(obs.columns).index(col))
        print(nc)
        return(nc)
    return(cols)




def preload(bl1,bl2,bl3,path = r'C:\Users\youm\Desktop\src',devmode=False):
    global SAVE
    global DEVMODE
    print(devmode)
    if devmode:
        SAVE = False
        DEVMODE = True
    if not devmode:
        path = checkChange(path,cat='folder to load from')
    print(len(os.listdir(path)),'files in folder')
    tstem = TSTEM
    if not devmode:
        tstem = checkChange(TSTEM,'stem of save files')
    for file in os.listdir(path):
        if  tstem == "_".join(file.split("_")[:-1]):
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0).astype(str)

    try:
        obs.name = obs.columns[-1]
    except UnboundLocalError:
        for file in sorted(os.listdir(path)):
            if ".csv" in file:
                print(file)
        print(TSTEM,": no files found! See above for list of .csvs in folder.")

        preload(9,9,9)
    return(df,obs,dfxy)


'''
menu functions ^
visu functions v
'''

def sortedMap(dfs,com=[],cat=''):
    #make sort-by-dendrogram function that adds "1- 2-" in front of any label so this has dendrogram option
    #then can run aggregate by leiden and run through here (after sorting highest level label (or any) by dendrogram)
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        return([],[])
    #    allcombs = getCats(obs,title='columns to sort by (in order)')
    #    return([],[allcombs])

    if 'sortedMap' in DONE:
        return(dfs,9)

    DONE.append('sortedMap')

    figtitle = ''
    dropNA = True
    dgram= False
    odf,oobs,oxy = dfs[0].copy(),dfs[1].copy(),dfs[2]
    oobs = oobs.astype(str)
    mpl.style.use('default')
    #sns.set(font_scale=2)


    allcombs = CATS


    print(allcombs,"allcombs!")

    sns.set(font_scale=3)
    if dropNA:  #ONLY DROPS NAN FROM FIRST CATEGORY e.g. if tumor subtype is first, it will make a map of tumor cells only
        col1 = CATS[0]
        nkey = oobs[col1].isna()
        nk2 = oobs[col1] == "nan"
        nkey = nkey | nk2
        df,obs,dfxy = odf.loc[~nkey,:],oobs.loc[~nkey,:],oxy.loc[~nkey,:]

    df,obs,dfxy = cm.zscorev(df,obs,dfxy)
    allutys = []
    for col in allcombs:
        utys = sorted(list(obs.loc[:,col].astype(str).unique()))
        allutys.append(utys)


    print(allutys,'all utys')
    data,colors = sortMap(df,obs,dfxy,allcombs,allutys)
    #print("\n\n",colors)
    vout = 10
    ax=sns.clustermap(data, vmin=-vout, vmax=vout, cmap='bwr',row_colors=colors,
                          yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                          row_cluster=False, col_cluster=False, colors_ratio = 0.01)

    #if SAVEfIGS:
        #plt.savefig(save(0,BX+"_single-heatmap",figtitle+" "+"_".join(cols),typ="png"),bbox_inches='tight')
    if SAVE:
        plt.savefig(saveF(0,"annotation heatmaps/",'_'.join(allcombs)),bbox_inches='tight')
    plt.show()
    #if dgram != False:
        #singHeatmap(df,obs,dfxy,tcols=tcols,title=figtitle+" d-gram")
    mpl.style.use('default')
    cols = allcombs
    fig = plt.figure(figsize=(4*len(cols),8))
    for j,col in enumerate(cols):
        mpl.style.use('default')
        print(j)
        ax = fig.add_subplot(1,len(cols),j+1)
        leg = {}
        utys = sorted(list(obs.loc[:,col].astype(str).unique()))
        #sns.set(font_scale=3/len(utys))

        for i,uo in enumerate(utys):
            if uo == "nan" or uo == "":
                leg[uo]="lightgray"
            elif uo[0].isdigit() and ('3: tumor' in utys or '3 tumor' in utys):
                uoi = int(uo[0]) - 1
                leg[uo]=allc.colors[uoi]
            else:
                leg[uo]=allc.colors[i]

        print(leg)
        kl = leg.keys()
        vl = leg.values()
        x = np.arange(len(kl))
        y = np.ones_like(x)
        ax.bar(x,y,color=vl)
        ax.set_xticks(x)
        ax.set_xticklabels(kl,rotation=90,size=min(20,160/len(utys)))
        #ax.title.set_text(col,rotation = 90)
        ax.set_ylabel(col)
        #ax.set_title(col)
        ax.set_yticks([])
        plt.tight_layout()
    if SAVE:
        plt.savefig(saveF(0,"annotation heatmaps/",'legend'+'_'.join(allcombs)),bbox_inches='tight')
    plt.show()
    return(odf,oobs,oxy)


def sortMap(df,obs,dfxy=9,cols=["no cols included"],allutys = None): #cols aka allcombs
    data,colors = [],[]
    while len(colors) < len(cols):
        colors.append([])
    print(cols,'cols')
    col = cols[0]
    utys = allutys[0]
    print(utys,'utys')
    primts = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","5 stromal"]
    for i,uo in enumerate(utys):
        print(i,uo)
        key = obs.loc[:,col]==uo
        #print(key.sum())
        if key.sum() == 0:
            print(obs.loc[:,col],'obs loc:',col,'has no values that ==',uo)
            continue
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        if uo == "nan" or uo == "":
            colors[0].append(pd.Series(np.full((sobs.shape[0]),'lightgray'),index=sobs.index))
        elif uo in primts:
            uoi = primts.index(uo)
            colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[uoi]),index=sobs.index))
        else:
            colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[i]),index=sobs.index))

        if len(cols) > 1:
            d1,c1 = sortMap(sdf,sobs,cols=cols[1:],allutys = allutys[1:])
            print('ran sortmap')
            data.append(d1)

            for j,cS in enumerate(c1):
                colors[j+1].append(cS)
        else:
            data.append(sdf)
            print(sdf,'sdf')
    try:
        data = pd.concat(data,axis=0)
    except Exception as e:
        print(e,'!!')
        print(utys,"no vals to concat in data",obs.loc[:,col])
        print(data,'data')
        input()
    c = []
    for ci in range(len(colors)):
        cs = pd.concat(colors[ci],axis=0,ignore_index=False)
        c.append(cs)
    colors = c
    return(data,colors)





def boxplot(dfs,com=[],cat=''):
    df,obs = dfs[0].copy(),dfs[1].copy()
    if len(com) == 0:
        vich = input("violin plot instead? (y): ")
        fich = input("show outliers? (y)")
        colors = getCats(obs,title='category to color by')
        try:
            ncols = int(input('boxplots per row in fig'))
        except:
            ncols = 6


        return([],[colors,vich,fich,ncols])


    colors,vich,fich,ncols = com[1],com[2],com[3],com[4]
    ch = list(obs.columns).index(cat)
    for color in colors:
        if color not in obs.columns:
            print('skipping',color)
            continue
        try:
            binCol = obs.columns[ch].astype(int)
        except:
            binCol = obs.columns[ch]
        try:
            obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
            try:
                obs[binCol+' temp']=obs[binCol+' temp'].astype(int)
            except:
                obs[binCol+' temp']=obs[binCol+' temp'].astype(float)
            print('int type bins')
        except Exception as e:
            print('str type bins',e)
            obs[binCol+' temp'] = obs[binCol]+'!'


        obs = obs.sort_values(binCol+' temp',axis=0) #THIS MESSES UP THE INDEX
        obs[binCol]=obs[binCol].astype(str)
        colCol = obs[color].sort_values()
        title = cat+' x '+color

        boxH(df,obs,binCol,colCol,vich,fich,title=title,corr=False,ncols=ncols)
    return()#9,[dfs]) #also doesn't need to return anything... temps are getting returned somehow, maybe axis messed up..



def boxH(df,obs,binCol,colCol,vich,fich,title='',corr = False,ncols=6):
    mpl.style.use('default')
    dfo = df.merge(obs,left_index=True,right_index=True).sort_values(binCol+' temp')
    ubins = list(obs.loc[:,binCol].unique())
    ucols = list(colCol.unique())
    if len(ubins) * len(ucols) == 1:
        print('skipping boxplot for',title)
        return([99,9],[])
    if len(ucols) == 1:
        colCol = None
    print(dfo.shape)
    #print(dfo)
    ofich = fich
    nrows = int(df.shape[1]/ncols)
    if df.shape[1] % ncols != 0:
        nrows += 1
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(int((len(ubins)*len(ucols))**.5)*ncols*1.5,3*nrows))
    for i,marker in enumerate(df.columns):
        ix = i % ncols
        iy = int(i/ncols)
        fich = ofich
        try:
            if np.quantile(df.loc[:,marker],.25) == np.quantile(df.loc[:,marker],.75):
                fich = 'y'
        except Exception as e:
            print(e,df.shape,marker)
        if vich == "y" and fich == "y":
            sns.violinplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=True, ax=ax[iy,ix])
        elif vich == "y":
            sns.violinplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=False, ax=ax[iy,ix])
        else:
            if fich == "y":
                sns.boxplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=True, ax=ax[iy,ix])
            else:
                sns.boxplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=False, ax=ax[iy,ix])
        if False:#corr: #correlation calculated between different colors iff 2 colors
            key = dfo.loc[:,binCol] == uBinCol[0] #NOT DEFINED
            pop1 = dfo.loc[key,marker]
            pop2 = dfo.loc[~key,marker]
            t_stat, p_val = ttest_ind(pop1,pop2)
            mstat,pval = mannwhitneyu(pop1,pop2)
            print('t-statistic:', t_stat)
            print('p-value:', p_val)
            print(mstat,"mstat")
            plt.title('t-test p-value: '+str(p_val)+'\nMann-Whitney p-value:'+str(pval))
        '''
        if i == 0:
            try:
                ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
            except:
                ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        '''
        if i == 0:
            handles, labels = ax[iy,ix].get_legend_handles_labels()

        if len(ucols) > 1:
            ax[iy,ix].get_legend().remove()
        ax[iy,ix].grid(visible=True)
        ax[iy,ix].tick_params(axis="x", labelrotation = 85)
        #plt.xticks(rotation = 85)

    fig.legend(handles, labels, loc='upper left',bbox_to_anchor=(1, 1),title=title.split(' x ')[-1])
    #plt.grid(visible=True)
    #plt.xticks(rotation = 85)
    plt.tight_layout()
    if SAVE:
        if vich:
            plt.savefig(saveF(0,"boxplots/",'violin_'+title),bbox_inches='tight')
        else:
            plt.savefig(saveF(0,"boxplots/",title),bbox_inches='tight')
    plt.show()
    return(99,9) #obs ind changed, doesn't need to return anything




def biomSortedMap(dfs,com=[],cat=''):
    global DONE
    df,obs,dfxy = dfs[0].copy(),dfs[1].copy(),dfs[2]
    if len(com) == 0:
        for i,col in enumerate(df.columns):
            print(i,col)

        chs = getCats(df,'markers to sort by')  #chs sorting
        #print(chs,'should be int')
        ch2,uch2 = obMenu(obs,'make different plots for each category in column:')
        ch2 = obs.columns[ch2]
        return([],[chs,ch2])
    if 'biomSortedMap' in DONE:
        return(dfs,9)

    DONE.append('biomSortedMap')
    chs,ch2 = com[1],com[2]
    ch2 = list(obs.columns).index(ch2)
    for ch in chs: #int for each biomarker
        if ch not in df.columns:
            continue
        print(ch,'ch in bsm')
        ch = list(df.columns).index(ch)
        uch2 = obs.iloc[:,ch2].unique()
        ch1s = []
        uch1s = []
        for cat in CATS:
            ch1s.append(list(obs.columns).index(cat)) #ch1s cats colorbars
            uch1s.append(obs.loc[:,cat].unique())
        #except:
        for uc2 in uch2: #for each sep fig
            key = obs.iloc[:,ch2] == uc2
            sobs = obs.loc[key,:]
            sdf = df.loc[key,:]
            biomSMH(sdf,sobs,dfxy,ch,ch1s,uch1s,title=uc2+'_'+'-'.join(CATS))
    return(dfs,9)

def biomSMH(df,obs,dfxy,ch,ch1s,uch1s,title = 'all cells'):

    #obs['toSort'] = df.iloc[:,ch]
    scol = df.columns[ch]
    obs[scol] = df.columns[ch]
    df = df.sort_values(df.columns[ch]).apply(ZSC)
    obs = obs.sort_values(scol)
    vout = 10
    sns.set(font_scale=2.5)

    rcolors = []
    for i,ch1 in enumerate(ch1s): #ch1s cats
        #rcolors.iloc[:,j] = obs.iloc[:,ch1].copy()
        rc = obs.iloc[:,ch1s[i]].copy() #need shape and name!, will all be colored over
        uch = uch1s[i]
        for j,uc in enumerate(uch):
            print(uc)
            key = obs.iloc[:,ch1] == uc
            print(key.sum(),'ks')
            #rcolors.loc[key,i] = allc.colors[j]
            rc.loc[key] = allc.colors[j]
        rcolors.append(rc.copy())
    rcolors = pd.DataFrame(rcolors).transpose()
    print(rcolors,'rcolors')
    ax=sns.clustermap(df, vmin=-vout, vmax=vout, cmap='bwr',row_colors = rcolors,
                          yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                          row_cluster=False, col_cluster=False, colors_ratio = 0.01)
    blank = ''
    for i in range(40):
        blank = blank+' '
    blank = '\n'+blank
    plt.title(blank+scol+blank+title)
    if SAVE:
        plt.savefig(saveF(0,"heatmaps/",scol+'    '+title),bbox_inches='tight')
    plt.show()
    return(df,obs,dfxy)


def showUmap(dfs,com=[],cat='doesnt matter to calculate umap once',ymin=0):
    global UMAP
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        #colors = CATS#UCATS#getCats(obs,'category to show in umaps:')
        return([],[])
    if 'showUmap' in DONE:
        print('umap already done')
        return(dfs,com)
    DONE.append('showUmap')
    colors = CATS


    mpl.style.use('default')
    #26obs = obs.astype(str)
    df = df.astype(float)
    obs = obs.astype(str)
    print(df.index)
    print(obs.index)
    #if input("replace 'no' with np.nan in columns with no nan entry (to color gray) (y)") == 'y':

    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.umap(adata)
    plt.rcParams['figure.figsize'] = 8, 8
    print(colors,'colors')

    for color in colors:
        '''
        naStrs = ["no","No","Nan",'nan',"NA","NAN","NaN"]
        keys = []
        for ns in naStrs:
            keys.append(obs.loc[:,color] == ns)
        keys = pd.concat(keys,axis=1)
        key0 = keys.sum(axis=1)
        #print(key0,"key0",key0.max())
        if key0.max() == 1:
            print('replacing no with nan',color)
            key1 = key0 != 0
            obs.loc[key1,color] = np.nan
        '''
        #cat=color
        #obs[color] = obs[color].astype(str)
        #print("0:show all or 1:select whic or 2:each unique value in own map?")
        for ch in range(2):
            if ch == 1:
                obs[color] = obs.loc[:,color].astype(str)
                uColors = sorted(list(obs[color].unique()))
                #for i,clr in enumerate(uColors):
                for i,uch in enumerate(uColors):
                    itm = []
                    clr = []
                    for j in range(len(uColors)):
                        itm.append(uColors[j])
                        if i == j:
                            clr.append("blue")
                        else:
                            clr.append("lightgray")
                    cd = dict(zip(itm,clr))
                    #fig,ax = plt.subplots()
                    sc.pl.umap(adata,color = color,palette=cd,show=False)
                    if SAVE:
                        plt.savefig(saveF(0,"umaps/"+color,color+'_'+uch),bbox_inches='tight')
                    plt.show()

            else:
               print('plotting all!')
               sc.pl.umap(adata,color = color,na_color='yellow',show=False)
               if SAVE:
                   plt.savefig(saveF(0,"umaps/"+color,color+'_all'),bbox_inches='tight')
               plt.show() #this was commented out yet everything worked... why...

        '''
        except Exception as e:
            print(type(e),e)
            break
        '''
        if not os.path.isdir(SPATH+'/umaps/expression'):
            for biom in df.columns:
                sc.pl.umap(adata,color=biom,vmin=np.mean(df[biom])-np.std(df[biom]),vmax=np.mean(df[biom])+2*np.std(df[biom]),color_map='viridis', show=False)
                if SAVE:
                    plt.savefig(saveF(0,"umaps/expression",biom),bbox_inches='tight')
                plt.show()
    return([df,obs,dfxy],9)


def barplot(dfs,com=[],cat='',showPercentageText=True):
    print(cat)
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    mpl.style.use('default')
    if len(com) == 0:
        ch,uch = obMenu(obs,'column to sort x axis by')
        binCol = obs.columns[ch]
        return([],[binCol])
    print('showing spatial')
    binCol = com[1]
    if binCol not in obs.columns:
        print('skipping',binCol,'for',cat)
    ch = list(obs.columns).index(binCol)



    obs = obs.astype(str)
    #rch = int(input("show actual number (0) or percentages (1)?"))
    try:
        obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
        obs[binCol+' temp']=obs[binCol+' temp'].astype(float).astype(int)
        print('int type bins')
    except:
        print('str type bins')
        obs[binCol+' temp'] = obs[binCol]+'!'

    obs = obs.sort_values(binCol+' temp',axis=0) #THIS MESSES UP THE INDEX
    #obs[binCol]=obs[binCol].astype(str)
    bins = sorted(list(obs.iloc[:,ch].unique()))
    print(bins,'sorted bins')

    for i,he in enumerate(obs.columns):
        print(i,he)
    ch2 = list(obs.columns).index(cat)

    colors = sorted(list(obs.iloc[:,ch2].unique()))
    colCol = obs.columns[ch2]
    a = np.zeros((len(bins),len(colors)))
    b = np.zeros((len(bins),len(colors)))
    c = b.copy()
    d = c.copy()
    #table = np.zeroes(len(bins),len(colors))
    nClrs = []
    for clr in colors:
        nClrs.append(obs.loc[obs.loc[:,colCol] == clr,:].shape[0])

    for i in range(len(bins)):
        Bin = obs.loc[obs[binCol]==bins[i],:] #sobs
        nBin = Bin.shape[0]
        for j in range(len(colors)):
            n = Bin.loc[Bin[colCol]==colors[j],:].shape[0] #tobs

            if nBin == 0:
                nBin = 1
            a[i,j] = n
            b[i,j] = n/nBin

    colorDict = {}
    for i in range(len(colors)):
        if colors[i] == 'yes':
            colorDict[colors[i]] = 'darkred'
        elif colors[i] == 'no':
            colorDict[colors[i]] = 'darkgray'
        elif str(colors[i]) == 'nan':
            colorDict[colors[i]] = 'lightgray'
        colorDict[colors[i]] = allc.colors[i]
    togg = True
    for array in [a,b,]: #c,d
        a = array.copy()
        fig, ax = plt.subplots(figsize=(15+len(bins)/5,10))
        shape = a.shape
        #print("a",a)
        height = np.zeros(shape[0])

        for i in range(shape[1]):

            try:
                AX = ax.bar(bins,a[:,i],label=colors[i],bottom=height,color=colorDict[colors[i]])
            except Exception as e:
                AX = ax.bar(bins,a[:,i],label=colors[i],bottom=height)
            if showPercentageText:
                for j,rect in enumerate(AX):
                    nh = a[j,i]#rect.get_height()
                    if nh >= .005:
                        plt.text(rect.get_x()+rect.get_width()/2,height[j]+nh/2,round(nh,2))
            #cont = ax.BarContainer()
            #labs = np.round(a[:,i],3)
            #print(labs)
            #ax.bar_label(cont, labels=labs, label_type='center')
            height += a[:,i]
        plt.xticks(rotation = 85)
        plt.xlabel(binCol)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
        if SAVE:
            if togg:
                plt.savefig(saveF(0,"barplot",binCol+' x '+cat),bbox_inches='tight')
                togg = False
            else:
                plt.savefig(saveF(0,"barplot",'norm '+binCol+' x '+cat),bbox_inches='tight')
                togg = True

        plt.show()
    return(9,9)




def spatialLite(dfs,com=[],cat='',ymin=0):
    #df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    mpl.style.use('default')
    if len(com) == 0:
        try:
            fsm = float(input('figsize modifier?'))
        except:
            fsm = 1
        colors = []
        #while True:
        #    cin = input('color to add in ordered list for all cats:')
        #    if cin == '':
        #        break
        #    colors.append(cin)
        return([],[colors,fsm])
    print('showing spatial')
    colors = com[1]
    fsm = com[2]
    nobs,nxy = dfs[1],dfs[2]
    ch1 = cat
    uch = sorted(list(nobs.loc[:,ch1].unique()))

    #colors = colors + allc.colors
    colors = allc.colors
    for scene in nobs.loc[:,"slide_scene"].unique():
        key=nobs["slide_scene"]==scene
        sobs = nobs.loc[key,:]
        #sdf = ndf.loc[key,:]
        sxy = nxy.loc[key,:]
        #ax.set_aspect('equal')
        #ax.legend(uch,colors,bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500*fsm,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500*fsm))
            print(max(sxy.iloc[:,1]),"max Y")
        except Exception as e:
            print(e,"error setting fig and ax",scene)
            print(sxy.isna().any(),"isna")
            fig,ax = plt.subplots()
        for i,ty in enumerate(uch):
            if ty[0].isdigit() and ('3: tumor' in uch or '3 tumor' in uch):
                uoi = int(uo[0]) - 1
                co = colors[uoi]
            else:
                co = colors[i]
            if ty == 'yes' or ty == 'Yes':
                co = 'darkred'
            elif ty == 'no' or ty == 'No':
                co = 'lightgray'
            elif ty == 'nan' or ty == 'NaN' or ty == 'NAN' or ty == 'Nan':
                co = 'whitesmoke'
            #print(sobs.columns[ch1])
            key1 = sobs.loc[:,ch1]==ty
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
        lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=ch1)#, scatterpoints=1, fontsize=10)
        try:
            for k in range(len(uch)):
                lg.legendHandles[k]._sizes = [30]
        except Exception as e:
            print(e,"index k:",k,len(uch),lg.legendHandles)
        plt.title(scene)
        if SAVE:
            plt.savefig(saveF(0,"spatial",scene+"_"+cat),bbox_inches='tight')
        plt.show()
    return(dfs,9)

def errorBar(dfs,com=[],cat='',ymin=0):
    if len(com) == 0:
        return([],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    sep = cat
    name = []
    for bx in sorted(list(obs.loc[:,sep].unique())):
        name.append(bx)
    name = '..vs..'.join(name)
    print(name)

    spacer = 2 * len(list(obs.loc[:,sep].unique()))
    fig,ax = plt.subplots(figsize = (df.shape[1]*spacer/8,3))
    colors = allc.colors

    for w,bx in enumerate(sorted(list(obs.loc[:,sep].unique()))):
        key = obs.loc[:,sep] == bx
        sdf = df.loc[key,:]

        means,sds = [],[]
        for biom in df.columns:
            means.append(np.mean(sdf.loc[:,biom]))
            sds.append(np.std(sdf.loc[:,biom]))

        for i in range(df.shape[1]):
            if i == 0:
                ax.errorbar(i+w/spacer-1/(2*spacer),means[i],yerr=sds[i],marker="o",ls='none',ecolor=colors[w],c=colors[w],label=bx)
            else:
                ax.errorbar(i+w/spacer-1/(2*spacer),means[i],yerr=sds[i],marker="o",ls='none',ecolor=colors[w],c=colors[w])
            #4*i+w-1/2
    ax.set(title=name)
    ax.grid(axis='x')
    ax.grid(axis='y')
    ax.legend(bbox_to_anchor=(.1, -.3))
    plt.xticks(np.arange(df.shape[1]),labels=df.columns, rotation='vertical', fontsize=7)
    if SAVE:
        plt.savefig(saveF(0,"errorbar plots",sep),bbox_inches='tight')
        #plt.savefig(save(0,'comparison-errorbarplots all cases',name,typ='png'),bbox_inches='tight')
    plt.show()
    return(dfs,9)


def hist(dfs,com=[],cat=''): #['Primary Celltype autoCellType res: 1.0','Primary Celltype Leiden_30_primariescluster autotype0.75']

    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        return([],[])
    print('HISTOGRAM BROKEN')
    return([df,obs,dfxy],[])
    colNames = [cat]
    for col in df.columns:
        cdf = df.copy()
        cobs = obs.copy()
        trimmedCol,key = trimExtremes(df.loc[:,col])
        cobs = cobs.loc[key,:]
        cdf = cdf.loc[key,:]
        for CN in colNames:
        #continue                                  # f!!!!!!!!!! !!! !!!!!!!!!!!!!!!!
            xs,ys = [],[]
            sCN = sorted(list(cobs[CN].unique()))
            for ty in sCN:
                tdf = cdf.loc[cobs[CN]==ty,:]
                tcol = tdf[col]
                #tcol = trimExtremes(tdf[col])
                x,y = makeHist(tcol,tdf.shape[0]/2)
                print(x,y,'xy0')
                xs.append(x)
                try:
                    ys.append(rollingAve(y,n=int(tdf.shape[0]/20)))
                except:
                    ys.append([0])
            fig,ax=plt.subplots()
            print(xs,ys,'xsys')
            for i in range(len(xs)):
                try:
                    x,y,c,ty = xs[i],ys[i],allc.amcolors[i],sCN[i]
                    print(x,y,'xy1')
                    ax.plot(x,y,c,label=ty, alpha=0.5)
                except:
                    print("hist can't use amcolors")
                    x,y,c,ty = xs[i],ys[i],allc.colors[i],sCN[i]
                    ax.plot(x,y,c,label=ty, alpha=0.5)
            ax.legend()
            ax.set_title(col+" "+CN)
            plt.ylabel("log2 cell counts")
            if SAVE:
                plt.savefig(saveF(0,"histogram_"+cat,col+"_"+CN),bbox_inches='tight')
            plt.show()
    return([df,obs,dfxy],[])

def rollingAve(l,n=4): #n in each direction
    newL = []
    for i in range(len(l)):
        nears = []
        for j in range(n):
            left = i - j - 1
            right = i + j + 1
            if  left >= 0:
                nears.append(l[left])
            if right < len(l):
                nears.append(l[right])
        #print(nears)
        newL.append(stat.mean(nears))
    return(newL)



def makeHist(x,nb,orientation="vertical"):
    mx,Mx = min(x),max(x)
    rx = Mx-mx
    if rx == 0:
        print("no cells in histogram!!!")
        return(x,x)
    sx = rx/nb
    binCts=[]
    bins = np.arange(mx,Mx+sx,sx)
    for i in range(1,len(bins)):
        key = x>=bins[i-1]
        key1 = x<bins[i]
        ss = x.loc[key&key1]
        if ss.shape[0] > 1:
            binCts.append(np.log10(ss.shape[0])/np.log10(2))
        else:
            binCts.append(0)
    y = np.arange(len(binCts))*sx+mx
    X = binCts
    if orientation == "vertical":
        return(y,X)
    else:
        return(X,y)


def trimExtremes(ser,quantile=.99):
    sS = ser.sort_values()
    nInd = sS.shape[0]
    cutoff = sS.iloc[int(nInd * quantile)]
    #print(cutoff,"!")
    key = ser < cutoff
    ser = ser.loc[key]
    return(ser,key)


def clusterMeans(df,obs,dfxy,ch):
    clusterA = []
    df,obs,dfxy=obCluster(df, obs, dfxy,ch)
    ucl = np.unique(obs.loc[:,'Cluster'].values)
    nClusters = len(ucl)
    clusterA = np.zeros((nClusters,df.shape[1]))
    print("ucl",ucl)
    for i,c in enumerate(ucl):
        cl = df.loc[obs['Cluster'] == c,:]
        try:
            markerMeans = np.mean(cl.values,axis = 0)
        except:
            markerMeans = np.mean(cl.values.astype(float),axis = 0)
        clusterA[i,:] = markerMeans
    #print(clusterA,"clusterA")
    cdf = pd.DataFrame(clusterA,index=ucl,columns = df.columns)
    return(cdf)

def obCluster(df,obs,dfxy,ch):
    obs = obs.astype(str)
    chob = obs.columns[ch]
    print(chob,'chob')
    clusters = obs[chob].copy()
    print(clusters,'clustrs')
    uobs = np.unique(obs[chob].values)
    for i,uo in enumerate(uobs):
        print(i,":",uo)
        clusters.loc[obs[chob]==uo] = uo
    obs["Cluster"] = list(clusters)
    return(df,obs,dfxy)


def heatmap(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        return([],[])
    ch = list(obs.columns).index(cat)
    if len(list(obs.iloc[:,ch].unique())) < 3:
        print('skipping heatmap for',cat)
        return([df,obs,dfxy],[])
    cdf = clusterMeans(df,obs,dfxy,ch)
    print(cdf.index,"cdfind!")
    #print(min(10+cdf.shape[0]/5,2**15/100))
    print(cdf)
    h = min(10+cdf.shape[0]/5,2**15/100)
    f, ax = plt.subplots(figsize=(20, h))
    #bbox = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    #width, height = bbox.width, bbox.height
    #height *= f.dpi
    #print(height,"HEIGHT")
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    ax.title.set_text(cat)
    if SAVE:
        plt.savefig(saveF(0,"cluster heatmap",cat),bbox_inches='tight')
    plt.show()

    cdf = cdf.apply(ZSC).fillna(0)
    print(cdf)
    f, ax = plt.subplots(figsize=(20, min(10+cdf.shape[0]/5,2**15/100)))
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    ax.title.set_text('zscored_'+cat)
    if SAVE:
        plt.savefig(saveF(0,"cluster heatmap","z_"+cat),bbox_inches='tight')
    plt.show()

    return([df,obs,dfxy],[])

if __name__ == "__main__":
    df,obs,dfxy = preload(9,9,9,devmode=True)
    main(df,obs,dfxy)