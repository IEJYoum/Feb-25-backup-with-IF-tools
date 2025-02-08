# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:16:50 2021

@author: youm
"""

#visium location: T:\visium\Isaac\JK-2024-04-03

import os
import numpy as np
import pandas as pd
import math
import IFprocessing7 as ifp
import IFvisualization2 as ifv
import cmifAnalysis50 as cm
import SVM5 as sv
import time
import scipy as sp
import matplotlib as mpl
import RAT2 as RAT
mpl.style.use('default')







'''
Reqd packages:
allcolors, orthoType7, orthogonal7, combat1, napari7, orthoType5, cmifAnalysis49, IFprocessing7, cmifAnalysis50, IFvisualization2, SVM5, torchCluster4, torchClusterTester3, torchCluster5, RAT2
'''


#DATAFOLDER = r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Analysis\KLF4_Project\figures IY\pipeline output\by slide ie csv ie TMA\BR301-119__data'
#DATAFOLDER = r'Y:\Youm'
#DATAFOLDER = r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA1\Segmentation\pTMA1-25_CellposeSegmentation'
#DATAFOLDER = r'\\graylab\BCC_Chin_Lab_RDS\ChinData\Cyclic_Analysis\cmIF_2023-09-12_VS-mPanc-KC\For_IY'
#DATAFOLDER = r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-09-13_MCF7-KLF4\Segmentation'
#DATAFOLDER = r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\Jan 6 11_bx1A\11Bx1-5_A_data'#'Y:\Youm'#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC' #r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-10-04_KLF4-2\Segmentation'#r'ChinData\Cyclic_Workflow\cmIF_2023-10-04_KLF4-2\Segmentation\BR301-121_CellposeSegmentation'
#DATAFOLDER =  r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data'  #r'T:\Cyclic_Workflow\cmIF_2023-09-13_MCF7-KLF4\Segmentation\MCF7k-7_CellposeSegmentation'#r'T:\Cyclic_Workflow\cmIF_2023-10-04_KLF4-2_R8\QC\R1-R8_data\patched registered'#os.getcwd() #r'W:\ChinData\Cyclic_Workflow\cmIF_2022-09-21_W22\RegisteredImages'#r'C:\Users\youm\Desktop\src\RNA data dmytro'
DATAFOLDER =  r'D:\Classified_CSV\Classified_CSV'
#r'W:\ChinData\Cyclic_Workflow\cmIF_2020-11-01_PDAC\Segmentation' #r'T:\Cyclic_Analysis\cmIF_2024-03-19_AT-G3-mMB'
#r'W:\ChinData\Cyclic_Analysis\WOO\Figures IY\all_2023\Data\R1-R2_Ecad-segmentation'#'W:\ChinData\Cyclic_Analysis\cmIF_2024-05-14_VS-mPancreatitis\data\subtracted'#r'T:\Cyclic_Workflow\cmIF_2023-04-07_pTMA1\Segmentation\pTMA1-25_CellposeSegmentation'#r'T:\Cyclic_Analysis\Orion_2023-11-15_PDAC\ome-tiff_export_MS\U54_TMA\flipped'#r'W:\ChinData\Cyclic_Workflow\cmIF_2023-08-04_WOO\Segmentation'#r'C:\Users\youm\Desktop\src'
#\\accsmb.ohsu.edu\CEDAR\archive\seq\JungsunKim\in_house_data\Visium\3011\3011_Adjacent_1\outs\filtered_feature_bc_matrix
#r'\\accsmb.ohsu.edu\CEDAR\archive/seq/JungsunKim/in_house_data/Visium' #
#r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-10-04_KLF4-2\Segmentation\BC-TMA1-14_CellposeSegmentation'     #r'\\graylab\BCC_Chin_Lab_RDS\ChinData\Cyclic_Analysis\cmIF_2021-05-03_PDAC\U54-TMA-9\Jenny\Liver_Lung_PDAC\data'  #r'T:\Cyclic_Workflow\cmIF_2023-05-11_KLF4_MCF7+Ms\Segmentation'


#r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\june 13\HTA9-14Bx1-7__data'#'Y:\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-15Bx1-5_CellposeSegmentation'
#r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\may 23\HTA9-14Bx1-7__data'#r'W:\ChinData\Cyclic_Analysis\cmIF_2022-03-xx_mTMA2\GM'#r'Y:\Cyclic_Workflow'#r"C:\Users\youm\Desktop\src"#r"Y:\Cyclic_Analysis\cmIF_2019-10-04_MB-TMA\Chin_2019-10" #r'Y:\Cyclic_Workflow\cmIF_2019-12-02_JP-TMA1\Segmentation'#r"Y:\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-14Bx1-7_CellposeSegmentation"
SAVEFOLDER = r"C:\Users\youm\Desktop\src" #r'C:\Users\youm\Desktop\src\transformernet_training'#
DATAFOLDER = DATAFOLDER.replace("\ "[0],"/")
SAVEFOLDER = SAVEFOLDER.replace("\ "[0],"/")
TSTEM = 'MIT_2'#'WOO_nan_all+09-2020_3'#'WOO_03'#'06_MCF_both'#'march_KLF_07'#'agg_klf4_neighborhoods_typedbyslide5'#'agg_klg4_neighborhoods_nocellnsidk'#'2223_b_mw_1000_cbt'#'2333_b_raw'#'3011_b_mw1000'#'rna_3011_both_nodep'#'3011_test'#'rna_3011_both'#'86_GL_bigobs'#"91_GL"#'Aaron_topics_3011_2'#'rna_3011_cl'#'rna_2223A_4'#'rna_2223D_0'# 'rna_2223_combined_2'#'rna_2223A_3'#'11Bx1_A manual_99'#'BR301_tumonly_2'#"Agg_KLF_94"#'W_Or_93'#"Agg_KLF_95"#"z95_MCF7"#"KLF-Nov2"#"BR301_96"#"BC_93_tum"#'89_MS'#'zzz95_U54_IY'  #'86_pTMA1_6_98'#'zzz95_U54_IY'
#agg_klg4_neighborhoods_typedbyslide4 has neighborhood info in df, removed for v5
#'zzz95_U54'#'92_MS'#'KLF_Nov1'#'zzz92_U54'#'94_MS'#'KLF_Nov1'#'z91_MCF7-7'#'pTMA1_6pts_1'#'91_pTMA1'#'z95_MCF7'#'z98_MCF7'#'90_pt4076_pTMA1'#'93_b11_4076_pTMA1'#'93_pTMA1'#'94_pTMA1'#'95_GL'#'z3_GL631'#'93_BR301'#'198_MCF7_blank'#'199_MCF7'#'zzz_hta14'#'PIPELINE_bx2_95'#'zzzzzzz_pipeline_refined_99'#'PIPELINE_92'#'PIPELINE_hta14_bx1_99'#'bx2_training_set'#'PIPELINE_bx2_95'#'PIPELINE_92'#'PIPELINE_hta14_bx1_99'  #'zzzzzzz_pipeline_refined_99 #'bx2_training_set'#
#rna_2223_A_0
#srun --pty --time=1-0 --mem=64G --gres=gpu:1 --partition=gpu bash -i
# Y:\  ==  /home/groups/graylab_share/Chin_Lab/ChinData
#'3011_b_top1000_tt'#

#89_MS has per-sample celltyping

#89_pTMA1 has placenta etc.
#add jennys new annots to 95u54 W:\ChinData\Cyclic_Analysis\cmIF_2021-05-03_PDAC\U54-TMA-9\Jenny\data\20230916_U54-TMA_LeidenCelltypes_PrimaryMet.csv
#zzz95_U54_IY has placenta and nan cells. zzz95_U54_IY2 is cleaned
#zzz95_U54 has data with NAN patient- however index is shuffled, check fix in def preload
#zzz94 has all jenny's nan removed
#zzz93 has all patient no control tissues
#z98 has ecad and cells and nan values

TPATH =  ''#r'C:\Users\youm\Desktop\src\BR MFC7 GL data pre 230808 pre vietnam/'
DEVMODE = True

os.chdir(SAVEFOLDER)

def main(dataFolder=DATAFOLDER,saveFolder=SAVEFOLDER):
    options = ["Import and clean data","load prepared data","use: "+TSTEM,'import rna data','load most recent save']
    functions = [buildDataFrame,load,preload,RAT.main,loadLast]
    df,obs,dfxy = menu(options,functions,esc = True)
    #print(list(df.columns))
    #print(df.shape,obs.shape,dfxy.shape)

    obs = cleanObs(obs)

    obs.name = obs.columns[-1]
    op = ["data editing","analysis","visualization","Support Vector Machine","old analysis tool"]
    fn = [loadingMenu,ifp.main,ifv.main,sv.main,cm.main]
    while True:
        dxys = dfxy.shape[0]
        df,obs,dfxy=menu(op,fn,df,obs,dfxy)
        if dfxy.shape[0] != dxys:
            for i in range(10):
                print('shape of data changed!')
            input('hit any key to continue')
        inp = input("quit?")
        if inp == "y" or inp == "":
            return(df,obs,dfxy)

def cleanObs(obs):
    for i in range(obs.shape[1]):
        try:
            obs.iloc[:,i] = obs.iloc[:,i].astype(str) #.astype(int) was here till removed for learning rate... will have to re-add for normal figs
        except:
            pass
    obs = obs.astype(str)
    return(obs)

def menu(options,functions,df=9,obs=9,dfxy=9,esc = False): #MANUAL MENU
    print("menu")
    while True:
        print("\n")
        for i,op in enumerate(options):
            print(i,":",op)
        try:
            print("send non-int when done (return df)")
            ch = int(input("number: "))
        except:
            return(df,obs,dfxy)
        if DEVMODE:
            df,obs,dfxy=functions[ch](df,obs,dfxy)
            if esc:
                return(df,obs,dfxy)
            print(all(obs.index==df.index),"all index the same")
            try:
                print(obs.columns)
            except:
                print("no obs")
            try:
                print(df.shape,"df shape")
            except:
                pass
        else:
            try:
                df,obs,dfxy=functions[ch](df,obs,dfxy)
                if esc:
                    return(df,obs,dfxy)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print('Failed!',e)




def loadingMenu(df=9,obs=9,dfxy=9):
    obs = obs.astype(str)
    op = ["save","drop cells with less than N% of data",
          "edit observations","drop columns based on key string","import biomarkers from datatable/observations from annotation table","combine obs",
          "save unique list of obs for making import table","rename column",
          "combine another prepared dataset (and handle mixed partitions)","handle mixed partitons in existing data",
          "scale data (z-score, etc)","autoclean NA values","edit observation labels","fillna","annotate cells that agree in other annotation categories"]
    fn = [save,dropCells,editObs,dropCols,importBiom,
          combineObs,saveObs,renCol,combineData,doPart,scale,autoClean,editLabels,fillNA,agreeThresh]
    df,obs,dfxy = menu(op,fn,df,obs,dfxy)
    print(df.shape,obs.shape,dfxy.shape)
    return(df,obs,dfxy)


def agreeThresh(df,obs,dfxy): #slow but designed to catch every case for if certain annots dont exist in cols, wrote quickly not optimized
    cols = cm.multiObMenu(obs,'columns whose annotations will be searched for agreement agree')
    print('note: annotations must be identical to count as agreement')
    outn = input('resulting column name? ')
    thresh = float(input('agreement threshold (0 combines all unique relevant annots, 1 = all columns must agree'))
    uch = []
    obs = obs.astype(str)
    for col in cols:
        uc = obs.loc[:,col].unique()
        for u in uc:
            if u == 'nan' or u == 'no':
                continue
            if u not in uch:
                if u == 'yes' or u == 'Yes' or u == 'YES' or u == 'Y' or u == 'y' or u == '+':
                    u = col+'_'+u
                uch.append(u)
    uch = sorted(uch)
    lcl = len(cols)
    outS = pd.Series('',index = obs.index)
    for i in range(obs.shape[0]):
        row = obs.loc[:,cols].iloc[i,:]
        for uc in uch:
            score = (row == uc).sum()/lcl
            print(score)
            if score >= thresh:
                #print(row,uc,score)
                outS.iloc[i] += uc +'_'
    print(uch,'uch')
    obs[outn] = outS
    return(df,obs.astype(str),dfxy)








def scale(df,obs,dfxy):
    op = ["z-score","log2","scaled raw (div by stdev)",'invert column (convert distance to proximity)']
    fn = [zscore, log2, scaledRaw, invertCol]
    df,obs,dfxy = menu(op,fn,df,obs,dfxy)
    return(df,obs,dfxy)


def invertCol(df,obs,dfxy,dropaft = True):
    icols = cm.multiObMenu(df,'columns to invert')
    try:
        offs = float(input('offset (new minimum value before inversion- 1 is default)'))
    except:
        offs = 1
    for col in icols:
        co = df.loc[:,col]
        mi = co.min()
        co = co - mi + offs
        df[col+' inverted'] = co
        if dropaft:
            df = df.drop(col)
    return(df,obs,dfxy)

def zscore(df,obs,dfxy):
    df = sp.stats.zscore(df,axis=0,nan_policy="omit")
    return(df,obs,dfxy)

def log2(df,obs,dfxy):
    df = np.log2(df)
    return(df,obs,dfxy)

def scaledRaw(df,obs,dfxy):
    stds = np.std(df,axis=0)
    df = df/stds
    return(df,obs,dfxy)


def doPart(df,obs,dfxy): #edited recently, malwina version doesn't have done = []
    toSkip = flexMenu('strings to ignore (leave all categories)')
    newDF = pd.DataFrame()
    toSkip += ['nuclei_','cell_','cytoplasm_']
    done = []
    for col in df.columns:
        switch = 0
        for ts in toSkip:
            if ts in col:
                newDF[col] = df.loc[:,col]
                switch = 1
                break
        if switch == 0:
            bn = col.split('_')[0]+'_'
            if bn not in done:#in list(df.columns):
                done.append(bn)
                toComb = []
                for col1 in df.columns:
                    if bn in col1:
                        toComb.append(df.loc[:,col1])
                if len(toComb) == 1:
                    maxes = toComb[0]
                else:
                    maxes = pd.DataFrame(toComb).max(axis=0)
                print(maxes)
                print(maxes.shape)
                newDF[bn] = maxes

    print(newDF.columns)
    return(newDF,obs,dfxy)

def doPart1(df,obs,dfxy):
    dShort=[]
    for col in df.columns:
        if df[col].isna().any():
            shortname = col.split("_")[0]
            if shortname not in dShort:
                dShort.append(shortname)

    for sCol in dShort:
        toComb = []
        for col in df.columns:
            if sCol == col.split("_")[0]:
                toComb.append(col)
        print("to combine:",sCol,toComb)
        if len(toComb) > 1:
            df=combinePart(df,["uclei","nuc"],toComb,"nuc",sCol)
            df=combinePart(df,["ellmem"],toComb,"cellmem",sCol)
            df=combinePart(df,["ucadj","ytopla","erinuc","cyto"],toComb,"cyto",sCol)
    print(df.columns[df.isna().any()])
    return(df,obs,dfxy)


def combinePart(df,partitionL,toComb,NName,sCol):
    print(toComb)
    sDF = pd.DataFrame(index=df.index)
    for biomarker in toComb:
        for name in partitionL:
            if name in biomarker:
                sDF[biomarker] = df[biomarker]
                continue
    if sDF.shape[1]>1:
        print(sDF.columns,"sdf cols\n")
        df.loc[:,sCol+"_"+NName+"_combined"]=sDF.max(axis=1)
    else:
        print("sDF only has one entry apparently", sDF.columns)
    return(df)


def doPartHierarchical(df,obs,dfxy):
    #KC:
    #Take this order to pick one partition per marker, except pERK.
    #cellmem2p25 > Cytoplasm > exp5 > perinuc5 for Ecad, HER2, EGFR, pARK, which values are in this order it they express.
    order = "cellmem > Cytoplasm > exp5 > perinuc".split(" > ")
    order.append("nuclei")
    print("order:",order)

    dShort=[]

    for col in df.columns:
        if df[col].isna().any():
            shortname = col.split("_")[0]
            if shortname not in dShort:
                dShort.append(shortname)

    for sCol in dShort:
        toComb = []
        for col in df.columns:
            if sCol == col.split("_")[0]:
                toComb.append(col)
        print("to combine:",sCol,toComb)
        if len(toComb) > 1:
            df[sCol+"_combined"]=np.nan
            for partition in order:
                print(partition)
                for biomarker in toComb:
                    if partition in biomarker:
                        c = toComb.pop(toComb.index(biomarker))
                        df.loc[df[sCol+"_combined"].isna(),sCol+"_combined"]=df[c]
                        print(toComb)
        if len(toComb) >0:
            print("ERROR: ",toComb," partition not included in order, adding as lowest priority")
            for biomarker in toComb:
                df.loc[df[sCol+"_combined"].isna(),sCol]=df[biomarker]

    return(df,obs,dfxy)



def combineData(df,obs,dfxy):
    df1,obs2,xy3 = load(9,9,9)
    D = [df,obs,dfxy]
    D1 = [df1,obs2,xy3]
    if input("handle mixed partitions? (y)") == "y":
        dShort = []
        d1Short = []
        for col in df.columns:
            dShort.append(col.split("_")[0])
        for col in df1.columns:
            d1Short.append(col.split("_")[0])

        for col in df.columns:
            if col not in df1.columns:
                #print(col,"tnp28 only")
                sCol = col.split("_")[0]
                if sCol in d1Short:
                    print(sCol)
                    df[sCol] = df[col]

        for col in df1.columns:
            if col not in df.columns:
                #print(col,"tnp28 only")
                sCol = col.split("_")[0]
                if sCol in dShort:
                    print(sCol)
                    df1[sCol] = df1[col]


    for i in range(3):
        d = D[i]
        print(d.shape)
        d1 = D1[i]
        d  = pd.concat([d,d1],axis=0)
        D[i] = d
        print(D[i].shape)
    print(D[0].columns[D[0].isna().any()])
    for co in D[0].columns:
        print(co,D[0][co].isna().sum())
    return(D[0],D[1],D[2])


def combineObs(df,obs,dfxy):
    tdf,nobs,txy = load(9,9,9)
    binds = obs.index.intersection(nobs.index)
    tdf,txy = 9,9
    if input("combine all? (does not overwrite same names) (y)") == "y":
        for col in nobs:
            if col not in obs:
                obs[col] = ""
                obs.loc[binds,col] = nobs[col]
    else:
        for col in nobs.columns:
            if input("include (/overwrite)"+col+"? (y)") == "y":
                obs[col] = ""
                obs.loc[binds,col] = nobs[col]
    return(df,obs,dfxy)

def importBiom(df,obs,dfxy):
    if input("import annotations from csv? (y)") == "y":
        return(importObs(df,obs,dfxy))
    return(impB(df,obs,dfxy))



def impB(df,obs,dfxy):
    ndf,tobs,txy = load(9,9,9)
    tobs,txy = 9,9
    for col in ndf:
        if input("import "+col+" ? (y)") == "y":
            df[col] = ""
            df.loc[ndf.index,col] = ndf[col]
    return(df,obs,dfxy)



def importObs(df,obs,dfxy):
    while True:
        file = input("path to file including extension")
        try:
            nobs = pd.read_csv(file)
            print(nobs,'nobs')
            break
        except Exception as e:
            print(e,"couldn't read file")
            if input("return to main menu? (y)") == 'y':
                return(df,obs,dfxy)
    ch,uch = obMenu(obs,title="obs column with matching values to new file")
    ch1,uch1 = obMenu(nobs,title="new column with matching values")
    print(sorted(uch),'old values')
    print(sorted(uch1),'new values- should have matches')
    toch = []
    for col in nobs.columns:
        print(col,nobs.loc[:,col].unique())
        if input("include column? (y)") == 'y':
            if col not in obs.columns:
                obs[col] = ""
                toch.append(col)
            else:
                obs[col+"_new"]  = ""
                nobs[col+"_new"]  = nobs.loc[:,col]
                toch.append(col+"_new")
    for uc in uch:
        if uc not in uch1:
            continue
        key = obs.iloc[:,ch] == uc
        for ncol in toch:
            nkey = nobs.iloc[:,ch1] == uc
            val = nobs.loc[nkey,ncol].values[0]
            obs.loc[key,ncol] = val
    return(df,obs,dfxy)



def importObs1(df,obs,dfxy):
    while True:
        file = input("path to file including extension")
        try:
            nobs = pd.read_csv(file)
            print(nobs,'nobs')
            break
        except Exception as e:
            print(e,"couldn't read file")
            if input("return to main menu? (y)") == 'y':
                return(df,obs,dfxy)
    ch,uch = obMenu(obs,title="obs column with matching values to new file")
    ch1,uch1 = obMenu(nobs,title="new column with matching values")
    print(uch,'old values')
    print(uch1,'new values- should have matches')
    nobs.index = nobs.iloc[:,ch1]
    toch = []
    for col in nobs.columns:
        print(col,nobs.loc[:,col].unique())
        if input("include column? (y)") == 'y':
            if col not in obs.columns:
                obs[col] = ""
                toch.append(col)
            else:
                obs[col+"_new"]  = ""
                nobs[col+"_new"]  = nobs.loc[:,col]
                print(nobs[col+"_new"],'new nobs')
                toch.append(col+"_new")
            print(obs.columns)
    print(obs.columns,toch)
    for uc in uch:
        if uc not in uch1:
            continue
        key = obs.iloc[:,ch] == uc
        for ncol in toch:
            print(nobs)
            val = nobs.loc[uc,ncol]
            print("               ",uc,ncol,'val:',val)
            print(obs.columns)
            obs.loc[key,ncol] = val
    return(df,obs,dfxy)


def renCol(df,obs,dfxy):
    ds = [df,obs,dfxy]
    for d in ds:
        while True:
            print(d.columns)
            ip = input("column to rename?")
            nn = input("new name")
            try:
                if nn in d.columns:
                    d
                d[nn] = d.pop(ip)

            except Exception as e:
                print("invalid renames",e)
            ch = input("edit another in dataframe? (y)")
            if ch == "":
                break
            elif ch[0] != "y" and ch[0] != "Y":
                break
    ch = input("sort dfs? (y)")
    if ch == "":
        return(ds[0],ds[1],ds[2])
    if ch[0] == "Y" or ch[0] == "y":
        for i,d in enumerate(ds):
            ds[i] = d.loc[:,d.columns.sort_values()]
        return(ds[0],ds[1],ds[2])
    return(ds[0],ds[1],ds[2])


def editLabels(df,obs,dfxy):
    ch,uch = obMenu(obs,"categorty to edit labels")
    obCol = obs.columns[ch]
    print("send blank to skip")
    for uc in uch:
        print(uc)
        nn = input("new label:")
        if nn != "":
            obs.loc[obs[obCol]==uc,obCol] = nn
    return(df,obs,dfxy)




def importMissing(df,obs,dfxy):
    d1,o1,x1 = buildDataFrame(9,9,9)
    for i in range(df.shape[0]):
        if df.iloc[i,:].isnull().sum()>0:
            ind = obs["index"].iloc[i]
            key = o1["index"] == ind
            newData = d1.loc[key,:]

            if newData.shape[0] > 0:
                print(newData,newData.shape)
                for col in df.columns:
                    if math.isnan(float(df[col].iloc[i]))>0:
                        if col in newData.columns:
                            #try:
                                df[col].iloc[i] = newData[col].values[0]
                            #except:
                                #print(ind,col,"no values")
    return(df,obs,dfxy)


def fillNA(df,obs,dfxy):
    if input("Use average based on mean? (y):") == "y":
        counts = df.isnull().sum(axis=1)
        #mdf = df.loc[counts>0,:]
        mobs = obs.loc[counts>0,:]
        ch,uch = obMenu(mobs,title="import average means based on")
        for c in uch:
            key = obs.iloc[:,ch]==c
            sdf = df.loc[key,:]
            means = np.nanmean(sdf.values,axis=0)
            for i,col in enumerate(sdf.columns):
                sdf[col] = sdf[col].fillna(means[i])
            df.loc[key,:] = sdf
            return(df,obs,dfxy)
    elif input("fill with numerical value? (y):") == "y":
        val = float(input("value to fill for all missing values:"))
        df = df.fillna(val)
        return(df,obs,dfxy)
    elif input('import value from other column (y)') == 'y':
        froms = input('string found in column with no nan: ')
        tos = input('string found in column with nan: ')
        for col in df.columns:
            if froms not in col:
                continue
            bn = col.split('_')[0]
            for col1 in df.columns:
                if tos not in col1 or bn not in col1:
                    continue
                print(col,col1,'filling')
                tfkey = df.loc[:,col1].isnull()
                df.loc[tfkey,col1] = df.loc[tfkey,col]

    return(df,obs,dfxy)

def flexMenu(title="String to include in list"):
    lis = []
    while True:
        ch=input(title+" (send blank when done): ")
        if ch == "":
            return(lis)
        lis.append(ch)



def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,col)
    ch = int(input(title))
    uch = obs[obs.columns[ch]].unique()
    return(ch,uch)


def dropCols(df,obs,dfxy):
    if input("include based on keystring instead? (y)") == 'y':
        df, obs, dfxy = inclCols(df,obs,dfxy)
        return(df,obs,dfxy)
    lis = [df,obs,dfxy]
    for k,d in enumerate(lis):
        print(list(d.columns))
        toRem = flexMenu(title="remove all columns containing these strings (end with ! for exact string, hit enter to skip)")
        print("\nbefore:\n",lis[k].columns)
        dr = []
        for col in d.columns:
            for t in toRem:
                if t in col:
                    dr.append(col)
                elif '!' in t:
                    if t[:-1] == col:
                        dr.append(col)
        lis[k] = tryDrop(d,dr)
        print("after:\n",lis[k].columns)
    for d in lis:
        print("\n",d.columns)
    return(lis[0],lis[1],lis[2])

def inclCols(df,obs,dfxy):
    lis = [df,obs,dfxy]
    for k,d in enumerate(lis):
        incl = []
        print(list(d.columns))
        toRem = flexMenu(title="include all columns containing these strings")
        if len(toRem) == 0:
            print("skipping!")
            continue
        print("\nbefore:\n",lis[k].columns)
        for col in d.columns:
            for t in toRem:
                if t in col and col not in incl:
                    incl.append(col)
        print("\n\n",incl,"\n\n")
        lis[k] = d.loc[:,incl]
    return(lis[0],lis[1],lis[2])




def countNA(df,obs,dfxy):
    uSlide = obs['slide'].unique()
    uBiom = df.columns.unique()
    outD = pd.DataFrame(index=uSlide,columns = uBiom)
    for s in uSlide:
        key = obs["slide"] == s
        sdf = df.loc[key,:]
        denom = sdf.shape[0]
        for biom in uBiom:
            try:
                num = sdf[biom].isna().sum()
                outD.loc[s,biom] = num/denom
            except:
                print(s,biom)
                #pass
    outD.to_csv("NAN counts.csv")
    return(df,obs,dfxy)

def autoClean(df,obs,dfxy): #duplicated in IFA4, IFV2 - except it takes DFs instead of df obs dfxy
    #df,obs,dfxy,cdf = DFs[0],DFs[1],DFs[2],DFs[3]
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
    return(df,obs,dfxy)


def dropCells(df,obs,dfxy):
    if input("countNA to .csv?") == "y":
        df,obs,dfxy = countNA(df,obs,dfxy)
    for co in df.columns:
        print(co,df[co].isna().sum()/df.shape[0]*100)
    print("missing percents)")
    while True:
        try:
            cho = int(input("drop 0:cells   1:columns(bioms) : "))
            ch = int(input("max missing % threshold integer (0 to drop all cells with missing values, 100 to keep all):"))
            break
        except Exception as e:
            print(e)
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
    else:
        counts = df.isnull().sum(axis=0)
        #print(counts,counts.shape,df.shape)
        Mx = df.shape[0]
        pts = (np.ones(counts.shape[0]) - counts/Mx)*100
        pts = pd.Series(pts)
        key = pts >= 100-ch
        df = df.loc[:,key]
    for co in df.columns:
        print(co,df[co].isna().sum()/df.shape[0]*100)
    print(counts,counts.shape,df.shape)
    if input("label edge distance?") == "y":
        if "edge_distance" not in obs.columns:
            print("add edge_distance column to obs")
        else:
            minD = float(input("minimum distance from edge to keep: "))
            key = obs["edge_distance"].astype(float) > minD
            print(key)
            print(obs.shape)
            obs["edge distance >"+str(minD)] = "false"
            obs.loc[key,"edge distance >"+str(minD)] = "true"
            #obs=obs.loc[key,:]
            #df = df.loc[key,:]
            #dfxy = dfxy.loc[key,:]
            print(obs.shape)
    return(df,obs,dfxy)


def buildDataFrame(bl1,bl2,bl3,unp=True):
    global DATAFOLDER
    DATAFOLDER = cm.checkChange(DATAFOLDER,'load from here?')
    if DATAFOLDER == '':
        DATAFOLDER = os.getcwd()
    print("build")
    DFs,names,goodStrs = getDFs()
    df = sortDFs(DFs,names,goodStrs)
    df,obs,dfxy = makeObs(df)
    if unp:
        obs = unpackObs(obs)
    return(df,obs,dfxy)

def saveObs(df,obs,dfxy):
    ch,uch = obMenu(obs,title="save unique entries in category:")
    pd.Series(uch).to_csv(input("filename? ")+".csv")
    return(df,obs,dfxy)

def importObs1(df,obs,dfxy):
    file = navigate("C:/Users/youm/.spyder-py3/src")
    file = pd.read_csv(file,dtype=object,index_col=0)
    file=file.values
    print(file)
    keys = file[:,0]
    print("keys:",keys)
    ch,uob = obMenu(obs,"obs to apply keys to")
    names = []
    #newEnts1 = []
    for i in range(file.shape[1]-1):
        newEnts = file[:,i+1]
        print(newEnts)
        #newEnts1+=list(newEnts)
        names.append(input("name for above set of obs: "))
    #di = pd.concat([pd.Series(keys),pd.Series(newEnts1)],axis=1)
    #print(di)
    for j,name in enumerate(names):
        obs[name] = "other"
        #for ob in uob:
        for i,okey in enumerate(keys):
            try:
                key = obs[obs.columns[ch]] == okey
                if not any(key):
                    print("no result for",okey)
                obs.loc[key,name] = file[i,j+1]
            except:
                print(name,okey,"does not have entry")
    return(df,obs,dfxy)





def unpackObs(obs):
    obs["index"] = obs.index
    while True:
        for i,col in enumerate(obs.columns):
            print(i,col)
        try:
            ch = int(input("split column number:"))
            print("example:",obs.iloc[0,ch])
            c = input("char to split with:")
        except:
            break
        newCols = obs.iloc[0,ch].split(c)
        for j,col in enumerate(newCols):
            print(col)
            name=input("type name or hit enter to discard: ")
            if name == "":
                continue
            else:
                try:
                    obs[name] = obs.iloc[:,ch].apply(lambda n: n.split(c)[j])
                except:
                    print(name,"could not convert")
    ch = input("manually add obs? (y for yes)")
    if ch == "1" or ch == "y" or ch == "Y":
        while True:
            print("\n\nAdding new category...")
            name = input("name of category (send blank to exit): ")
            if name == "":
                break
            for i,col in enumerate(list(obs.columns)):
                print(i,col)
            try:
                print("recently added sorted llist 4 below- untested \nbeware!")
                ch = int(input("apply based on category number:"))
            except:
                break
            uobs = sorted(list(obs.iloc[:,ch].unique()))
            try:
                obs.loc[:,name]
            except:
                obs[name] = ""
            print('send "." to copy label from based-on-category')
            for uo in uobs:
                print("\n",uo)
                try:
                    curr = obs.loc[obs.iloc[:,ch]==uo,name].iloc[0]
                    print("current:",curr)
                except:
                    print("could not display current name")
                    curr = 'is this error possible?'
                annot = input("enter annotation:")
                if annot == "":
                    continue
                elif annot == '.':
                    print('using:',uo)
                    obs.loc[obs.iloc[:,ch]==uo,name] = uo
                else:
                    obs.loc[obs.iloc[:,ch]==uo,name] = annot
    while True:
        try:
            ch1,uch = obMenu(obs,"observation 1 to combine")
            ch2,uch = obMenu(obs,"observation 2 to combine")
            obs[obs.columns[ch1]+"_"+obs.columns[ch2]] = obs.iloc[:,ch1]+"_"+obs.iloc[:,ch2]
        except:
            break

    obs = obs.astype(str)
    return(obs)



def loadLast(bl1,bl2,bl3):
    global TSTEM
    for file in sortByTime(os.listdir(SAVEFOLDER)):
        if 'df.csv' in file:
            TSTEM = '_'.join(file.split('_')[:-1])
            print(TSTEM)
            return(preload(9,9,9))

def sortByTime(files):
    print(files)
    times = []
    for f in files:
        times.append(os.path.getmtime(SAVEFOLDER+'/'+f))
    sortd = []
    for i in range(len(files)):
        mind = times.index(max(times)) #min gets oldest
        sortd.append(files[mind])
        files = files[:mind]+files[mind+1:]
        times = times[:mind]+times[mind+1:]

    return(sortd)


def preload(bl1,bl2,bl3,path = TPATH):
    if path == "none" or path == "":
        path = SAVEFOLDER
    print(path)
    for file in os.listdir(path):
        if TSTEM == "_".join(file.split("_")[:-1]):
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0)
    obs.name = 'slide_scene'#obs.columns[-1]
    obs = obs.loc[list(df.index),:]
    dfxy = dfxy.loc[list(df.index),:]
    print(all(obs.index==df.index),"all index the same")
    print(df.index,obs.index)
    print(obs,'obs!')
    return(df,obs,dfxy)

def load(bl1,bl2,bl3,path = "none"):
    if path == "none":
        path = SAVEFOLDER
    print(path)
    #"C:/Users/youm/.spyder-py3/src"
    while True:
        path = navigate(path,text="select dataframe to load")
        print(path,"out of navigate")
        if path is list:
            print("please select specific file")
            continue
        if not os.path.isdir(path):
            break

    name = "_".join(path.split("_")[:-1])
    #print(name)
    name=name.split("/")[-1]+"_"
    #print(name,path)
    path = "/".join(path.split("/")[:-1])
    for file in os.listdir(path):
        fn = "_".join(file.split("_")[:-1])
        fn=fn.split("/")[-1]+"_"
        if fn == name:
            print(file)
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0)

    #ser = pd.Series(df.index).apply(lambda x: x.split('.1')[0])
    #df.index = ser
    #obs.index = ser
    #dfxy = dfxy.loc[df.index,:]
    return(df,obs,dfxy)


def editObs(df,obs,dfxy):
    ch = input("re-organize all obs? (y) :")
    if ch == "1" or ch == "y":
        df,obs,dfxy=makeObs(pd.concat([df,obs,dfxy],axis=1))
    obs=unpackObs(obs)
    return(df,obs,dfxy)

def makeObs(df):
    df,dfxy = splitDF(df,"X and Y coordinate columns")
    df,obs = splitDF(df,"Observation columns")
    df = makeDtype(df,dtype=float)
    obs = makeDtype(obs,dtype=str)
    dfxy = makeDtype(dfxy,dtype=float)
    for d in [df,obs,dfxy]:
        print(d,d.shape)
    return(df,obs,dfxy)


def makeDtype(df,dtype=str):
    print('MAKE DTYPE BROKEN')
    return(df)
    print("\n",dtype)
    for i in range(df.shape[1]):
        try:
            df.iloc[:,i] = df.iloc[:,i].astype(dtype)
        except:
            u = df.iloc[:,i].unique()
            if len(u) < 20:
                new = np.arange(len(u)).astype(dtype)
                print(u,new)
                for j,un in enumerate(u):
                    key = df.iloc[:,i] == un
                    df.iloc[:,i].loc[key] = new[j]
                df.iloc[:,i] = df.iloc[:,i].astype(dtype)
            else:
                df.iloc[:,i] = np.zeros(df.shape[0]).astype(dtype)
    return(df)


def splitDF(df,titleStr="new"):
    print(titleStr)
    for i,col in enumerate(df.columns):
        print(i,col)
    newDF = []
    dropList = []
    while True:
        try:
            print('it seems like using a range after getting here via reorg-all-obs causes df shape mismatch downstream')
            ch = input("column to split off into "+titleStr+": ")
            try:
                ch = int(ch)
            except:
                try:
                    ch = eval(ch)
                except:
                    chl = ch.split(":")
                    ch = range(int(chl[0]),int(chl[1]))
        except ValueError:
            break
        print(ch)
        newDF.append(df.loc[:,pd.Series(df.columns).iloc[ch]])
        nl =  df.columns[ch]
        if type(nl) == str:
            nl = [df.columns[ch]]
        else:
            nl = list(df.columns[ch])

        dropList+=nl
    try:
        newDF = pd.concat(newDF,axis=1)
        df = tryDrop(df,dropList)
    except ValueError as e:
        print(e)
        newDF = pd.DataFrame(index=df.index,data=df.index)
    return(df,newDF)

def tryDrop(df,dropList):
    for colName in dropList:
        try:
            df = df.drop([colName],axis = 1)
        except Exception as e:
            print(e,colName)
            #print(colName,'not in dataframe')
    return(df)

def sortDFs(DFs,names,goodStrs):

    #int(DFs[0].index[0])
    #print("integer index labels found")
    print("filenames",names)
    print(DFs[0].index[0],'sample index label')
    cheee = 0
    if input("add file names to index? (y)") == 'y':
        if input("reset index as simple integers? (y)") == "y":
            cheee = 1

        spli = input("character to split filenames with?")
        for i,name in enumerate(names):
            print(name,"nam!")
            if cheee:
                DFs[i].index = np.arange(DFs[i].shape[0])+1
            sind = pd.Series(DFs[i].index).astype(str)
            sind = name.split(spli)[0]+spli + sind
            print(sind,"sind")
            DFs[i].index = sind

    DFs,names = pd.Series(DFs),pd.Series(names)
    i = 0
    print(names)
    switch = 0
    if len(goodStrs) == names.shape[0]:
        DF = pd.concat(list(DFs),axis=1)
        return(DF)
    for ch in goodStrs:
        print("\n",ch)
        #key = names.str.contains(ch) #this is wrong sometimes- failed with s. and es.
        key = []
        for name in names:
            if ch in name:
                key.append(True)
            else:
                key.append(False)
        key = pd.Series(key)
        print(key)
        sDFs = DFs.loc[key]
        #print(np.array(sDFs).shape)
        if sDFs.shape[0] == 0:
            continue

        if i == 0:
            DF = pd.concat(list(sDFs),axis=0)
            print("first",DF.shape)
            i = 1
        else:
            DF2 = pd.concat(list(sDFs),axis=0)
            print(DF.index,DF2.index)
            if DF.shape[0] > DF2.shape[0]: #new feb 2024
                DF = DF.loc[DF2.index,:]
            else:
                DF2 = DF2.loc[DF.index,:]
            DF = pd.concat([DF,DF2],axis=1)
            #DF = DF.merge(pd.concat(list(sDFs),axis=0),how="outer",left_index=True,right_index=True)
            print(DF.shape)
    print("DF Final:",DF.shape)
    return(DF)





def getDFs():
    global DATAFOLDER
    DFs = []
    names = []
    goodStrs = []
    paths = []
    while True:
        path = getFile(DATAFOLDER)
        #print(type(path))
        if path == 'done':#isinstance(path,type(None)):
            break
        elif type(path) == list:
            path = path[0]
            goodStrs1 = flexMenu(title="add string that file name must contain along axis (e.g. centroid), send blank when done:")
            goodStrs += goodStrs1
            for f in sorted(os.listdir(path)):
                cond=0
                for gs in goodStrs1:
                    if gs in f:
                        cond=1
                if ".csv" in f and cond!=0:
                    try:
                        ndf = pd.read_csv(path+"/"+f,index_col=0)
                        #print(ndf.shape[1],"shape1")
                        if ndf.shape[1] < 2:
                           ndf = pd.read_csv(path+"/"+f,dtype=object,sep=" ")
                           print(ndf.shape,"shape!")
                        DFs.append(ndf)
                        names.append(f)
                    except:
                        print("error processing",f)
            return(DFs,names,goodStrs)
        else:
            #goodStrs.append(path.split("/")[-1])
            paths.append(path)
    for path in sorted(paths):
        print(path)
        names.append(path.split("/")[-1])
        try:
            ndf = pd.read_csv(path,index_col=0)
        except Exception as e:
            print(e)
            continue
        if ndf.shape[1] < 2:
           ndf = pd.read_csv(path,dtype=object,sep=" ")
           print(ndf.shape,"shape!")
        DFs.append(ndf)
    goodStrs1 = flexMenu(title="add string that file name must contain along axis (e.g. centroid), send blank when done:")
    goodStrs += goodStrs1
    print(names,"loaded files")
    for i,n in enumerate(names):
        print(DFs[i].shape,'shape')
        if input(n+' transpose? (y)') == 'y':
            DFs[i] = DFs[i].transpose()


    return(DFs,names,goodStrs)


def getFile(folder):
    path = folder
    if not os.path.exists(path):
        print("error: ",path," is invalid path")
        path = input("manually select path")
        return(getFile(path))
    try:
        while os.path.isdir(path):
            try:
                path=navigate(path)
            except Exception as e:
                print(e,"error 2")
                path = folder
            if type(path)==list:
                return(path)
    except TypeError as e:
        print(e,"error 1, probably no connection to folder")
    return(path)

def navigate1(path,text="send blank to go back to parent directory, send 'all' to return entire folder, 'done' to return what's loaded"): #auto
    global COMPOS
    folder = sorted(os.listdir(path))
    for i,thing in enumerate(folder):
        print(i,thing)
    print("\n"+text)
    #ch = input("access which number?")
    ch = COMMANDLIST[COMPOS]
    COMPOS += 1
    if ch == "":
        print("going to parent directory")
        plis = path.split("/")
        plis = plis[:-1]
        path = "/".join(plis)
    elif ch == "all":
        return([path])
    elif ch == "quit" or ch[0] == "q" or ch == "done":
        return('done')
    else:
        ch = int(ch)
        path = path+"/"+folder[ch]
    print(path)
    return(path)

def navigate(path,text="send blank to go back to parent directory, send 'all' to return entire folder, 'done' to return what's loaded"): #manual
    of = sorted(os.listdir(path))
    folder = []
    for thing in of:
        if not os.path.isdir(path+"/"+thing):
            if ".csv" not in thing:
                continue
        folder.append(thing)
    for i,thing in enumerate(folder):
        print(i,thing)
    print("\n"+text)
    ch = input("access which number?")
    if ch == "":
        print("going to parent directory")
        plis = path.split("/")
        plis = plis[:-1]
        path = "/".join(plis)
    elif ch == "all":
        return([path])
    elif ch == "quit" or ch[0] == "q" or ch == "done":
        return('done')
    else:
        try:
            ch = int(ch)
            path = path+"/"+folder[ch]
        except:
            for i,opt in enumerate(folder):
                if ch in opt and ".csv" in opt:
                    print(i,":",opt)
            ch = input("access which number?")
            path = path+"/"+folder[int(ch)]
    print(path)
    return(path)

def multisave(df,obs,dfxy):
    filename = input("prefix: ")
    ch,uch = obMenu(obs,title="category to divide along to save as .csvs")
    if len(uch) > 50:
        print("error, trying to save more than 50 .csvs")
        return(df,obs,dfxy)
    for uc in uch:
        key = obs.iloc[:,ch] == uc
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        sxy = dfxy.loc[key,:]
        sdf.to_csv(filename+"_"+uc+"_df.csv")
        sobs.to_csv(filename+"_"+uc+"_obs.csv")
        sxy.to_csv(filename+"_"+uc+"_dfxy.csv")
    return(df,obs,dfxy)

def save(df,obs,dfxy,filename=None):
    if not filename:
        filename = input("filename: ")
    if len(filename) < 2:
        for file in sortByTime(os.listdir(SAVEFOLDER)):
            if 'df.csv' in file:
                filename='_'.join(file.split('_')[:-1])
                print('overwriting',file,filename)
                ch = input('overwrite most recent save? (y)')
                if ch != '' and ch != 'y':
                    return(save(df,obs,dfxy))
                break

    df.to_csv(filename+"_df.csv")
    obs.to_csv(filename+"_obs.csv")
    dfxy.to_csv(filename+"_dfxy.csv")
    return(df,obs,dfxy)


if __name__ == "__main__":
    main()#r"Y:\ChinData\Cyclic_Analysis\20210413_AMTEC_Analysis")


    '''

            #df[sCol+"_combined"]=np.nan
            sDF = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker:
                    sDF[biomarker] = df[biomarker]
            print(sDF.columns)
            df.loc[:,sCol+"_nuc_combined"]=sDF.max(axis=1)
            sDF = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker:
                    sDF[biomarker] = df[biomarker]
            print(sDF.columns)
            df.loc[:,sCol+"_nuc_combined"]=sDF.max(axis=1)



            sDF1 = pd.DataFrame(index=df.index)
            for biomarker in toComb:
                if "uclei" in biomarker or "perinuc" in biomarker or "nucadj" in biomarker or "exp5" in biomarker or "ellmem" in biomarker:
                    continue
                else:
                    sDF1[biomarker] = df[biomarker]
            print(sDF1.columns)
            df.loc[:,sCol+"_cyto_combined"]=sDF1.max(axis=1)



    '''
