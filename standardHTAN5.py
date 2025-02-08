
'''
op = ["start over with raw data","log2","scale from -1 to 1", "z-score","elmarScale","trim outliers",
      "make control TMA sample sizes the same","combat",
      "apply TMA combat to other dataset","equalizeBiomLevel","adjust for negative values",
      "save to csv", "pick subset of data", "manually threshold",
      "cluster by obs catagory","Leiden cluster","GMM cluster","aggregate",
      "manually celltype random training set","auto-cell-type",
      "convert df to fractions in obs categories","convert to superbiom-only df",
      "remove non-primary biomarkers","calculate biomarker expression in region around each cell",
      "count label fractions in neighborhood","select ROI","remove cells expressing certain biomarker combinations"]
fn = [revert,log2,scale1,zscore,elmarScale,outliers,equalizeTMA,combat,TMAcombat,equalizeBiomLevel,remNegatives,save,pick,
      manThresh,obCluster,leiden,gmm,aggregate,celltype,autotype,countObs,superBiomDF,
      onlyPrimaries,regionAverage,neighborhoodFractions,roi,simulateTherapy]

op = ["heatmap","cluster-bar plot","bar plot","box plot","correlation matrix","scatterplot","umap","show spatial","scanpy visuals"]#,"send cluster df to processing"]
fn = [heatmap,clusterbar,barplot,boxplot,correlationMatrix,scatterplot,showUmap,spatial,scanpyv]
'''
import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
import allcolors as allc
import matplotlib as mpl
import os
import time
from sklearn.cluster import KMeans
import skimage
from skimage import io
from random import randint
import tifffile
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import BayesianGaussianMixture as GMM
import PIL
import SVM6 as sv

import warnings
warnings.simplefilter(action='ignore')

'''
['Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HER2A-22_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-10Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-10Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-11Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-11Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-12Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-12Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-13Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-13Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-16Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-16Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-17Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-17Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-18Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-18Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-19Bx5-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-7Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-8Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-8Bx2-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-9Bx1-5_CellposeSegmentation', 'Y:\\Cyclic_Workflow\\cmIF_2022-06-20_HTA9-7-A\\Segmentation/HTA9-9Bx2-5_CellposeSegmentation']
['HER2A_A', 'HTA9-10Bx1_A', 'HTA9-10Bx2_A', 'HTA9-11Bx1_A', 'HTA9-11Bx2_A', 'HTA9-12Bx1_A', 'HTA9-12Bx2_A', 'HTA9-13Bx1_A', 'HTA9-13Bx2_A', 'HTA9-16Bx1_A', 'HTA9-16Bx2_A', 'HTA9-17Bx1_A', 'HTA9-17Bx2_A', 'HTA9-18Bx1_A', 'HTA9-18Bx2_A', 'HTA9-19Bx5_A', 'HTA9-7Bx2_A', 'HTA9-8Bx1_A', 'HTA9-8Bx2_A', 'HTA9-9Bx1_A', 'HTA9-9Bx2_A']
'''


#import scimap
savedir = r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\Feb 8'    #'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\primary + celltyping including ki67 pcna area eccentricity'
#r'C:\Users\youm\Desktop\src\zzzzzzzzzzz_current\savedir'
#folders =  ["Y:\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-14Bx1-7_CellposeSegmentation"]   #r"Y:\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation/HTA9-14Bx2-5_CellposeSegmentation",
#FDIRS = [r'Y:\Cyclic_Workflow\cmIF_2022-06-20_HTA9-7-A\Segmentation',r'Y:\Cyclic_Workflow\cmIF_2022-08-08_HTA9-7-B\Segmentation']
folders = ['Y:\Cyclic_Workflow\cmIF_2022-06-20_HTA9-7-A\Segmentation\HTA9-11Bx1-5_CellposeSegmentation']
bxs = ['11Bx1-5_A']
'''
tag = 'A'
for FDIR in FDIRS:
    for item in sorted(os.listdir(FDIR)):
        folders.append(FDIR+'/'+item)
        bxs.append('-'.join(item.split('-')[:-1])+'_'+tag)   #MAKE SURE TO CHANGE THIS FOR EACH CHANNEL _A _B
    tag = 'B'

check standardHTAN5py.py for the version that processed the whole big batch
'''
print(folders)
print(bxs)


SP = ''
SPI = 0#2  #set to 0 for first segpath
#folders = folders[2:]
#bxs = bxs[2:]
#folder = r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
#stem = folder+"14_both"
#stems =[stem]#["HTA14f"]#,"HTA14faf","HTA15f","HTA15faf"] #'hta1415Baf'#"0.93 TNP-TMA-28"#"HTA14f"
OUT = []

BX = ""
oldData = []
SAVEfIGS = True#

BADMARKERl =  []#["CD3_","CD8","HER","CD20"]

def main2():
    global BX
    pairs = getPairs()
    print(pairs)
    for pair in pairs:
        BX = pair[0]+"..vs.."+pair[1]
        dfs = []
        for i in range(2):
            fold = pair[i]+"_data"
            for file in os.listdir(savedir+'/'+fold):
                print("Loading",file)
                dfs.append(pd.read_csv(savedir+'/'+fold+'/'+file,index_col=0))
        df = pd.concat(dfs,axis=0)
        df,obs,dfxy = splitDF(df)
        visu1([df,obs,dfxy])


def visu1(dfs):
    df,obs,dfxy=dfs[0],dfs[1],dfs[2]
    mpl.style.use('default')
    showUmap(df,obs,dfxy,keySs = ['SVM_','autoCell',"_high","slide"])
    obs["type_slide"] =  obs.loc[:,'SVM_primary'] + obs.loc[:,"slide"]
    callBox(df,obs,dfxy,bins = ['type_slide','slide_scene'])
    callMap(df,obs,dfxy,slide=True)


def splitDF(df):
    cols = list(df.columns)
    eind = cols.index('Ecad_negative')
    dcols = cols[:eind]
    ocols = cols[eind:-2]
    xcols = cols[-2:]
    ndf = df.loc[:,dcols].astype(float)
    obs = df.loc[:,ocols]
    dfxy = df.loc[:,xcols]
    print(df,obs,dfxy)
    #obs.to_csv("test.csv")
    #1/0
    return(ndf,obs,dfxy)


def getPairs():
    pairs = []
    for bx in bxs:
        bL = bx.split('_')
        nam = bL[0].split('B')[0]
        print(bL,nam)
        for bx1 in bxs:
            if bx1 == bx:
                continue
            bL1 = bx1.split('_')
            nam1 = bL1[0].split('B')[0]
            if nam==nam1 and bL[-1] == bL1[-1]:
                pairs.append(sorted([bx,bx1]))
    return(pairs)


'''
'''

def main():
    input("ONLY USING SCENE 2. Hit enter to confirm recipt.")
    elog = []
    global BX
    global SP
    xNames = ["DAPI_X","DAPI_Y"]
    obNames = ["slide","celli","Ecad_n"]
    ii = 0 #set to 0 for start
    while ii < len(folders):
        fold = folders[ii]
        BX = bxs[ii]
        #SP = segPaths[ii]
        ii += 1
        for file in os.listdir(fold):
            if "patched" in file and "regis" in file and "subtr" not in file:
                if "_A" in BX:
                    if "DAPI11" not in file:
                        print('not running:',file)
                        continue
                fn = file.split("_")[1]+"_"
                print(fn)
                OUT.append([fold,file])
                DF = pd.read_csv(fold+"/"+file,index_col=0)
                dfs = [[],[],[]]
                m = 1
                for col in sorted(list(DF.columns)):
                    if checkL(col,xNames):
                        dfs[2].append(DF.loc[:,col]*m)
                    elif checkL(col,obNames):
                        dfs[1].append(DF.loc[:,col])
                    else:
                        try:
                            se = DF.loc[:,col].astype(float)
                            dfs[0].append(se)
                        except Exception as e:
                            print(e)
                for i in range(3):
                    #print(dfs[i][0].shape,DF.shape)
                    dfs[i] = pd.DataFrame(pd.concat(dfs[i],axis=1),index=DF.index)
                    #print(dfs[i].columns)

                try:
                    dfs = processing(dfs)
                    print(dfs[1].columns)
                    dfs = analysis(dfs)
                    spatialLite(dfs[1],dfs[2],colNames = ['receptors SVM_phenotype',"receptors expressed"])

                    #print(dfs[1].columns)
                    #dfs[0] =dfs[3] use to make plots on primary markers only
                    foln = BX+"_data"
                    Dfs = pd.concat(dfs,axis=1)
                    if SAVEfIGS:
                        save(Dfs,foln,fn+'_1',typ="csv")
                    #visu(dfs)
                    df,obs,dfxy=dfs[0],dfs[1],dfs[2]
                    callMap(df,obs,dfxy)
                    #print("                    NO VISU")
                    print(BX)
                #print(dfs[1].columns)
                except Exception as e:
                    elog.append([BX,e])
    if len(elog) > 0:
        elog = pd.DataFrame(elog)
        elog.to_csv('error log1.csv')








def visu(dfs):
    mpl.style.use('default')
    df,obs,dfxy=dfs[0],dfs[1],dfs[2]
    #print(obs.columns)
    #'''
    silh(df,obs,dfxy)

    try:
        pass
        #saveOmeTiff(obs)
    except Exception as e:
        print(e,"OME TIFF FAILED!!")
    #return()
    oobs = obs.copy()
    dit,doot,dat = showUmap(df,obs,dfxy)
    obs = oobs.copy()
    #obs = obs.astype(str)
    zdf,sobs,sxy = zscorev(df,obs,dfxy)
    #
    boxplot1(zdf,sobs,sxy)
    callBox(df,obs,dfxy)
    #'''
    try:
        hist(df,obs,dfxy)
    except:
        pass
    callBar(df,obs,dfxy)
    #'''

    spatialLite(obs,dfxy)
    callMap(df,obs,dfxy)
    #'''
    print(obs.columns)

def analysis(dfs):
    df,obs,dfxy=dfs[0],dfs[1],dfs[2]

    df,obs,dfxy = autotype(df,obs,dfxy,res=1.0) #make first for obs ordering with res = 1
    ndfs = reorder(dfs,primaries=True)
    prdf,probs,prxy = ndfs[0],ndfs[1],ndfs[2]
    probs = probs.iloc[:,0:1]
    xdf,probs,xdfxy = autoleiden(prdf,probs,prxy,30)
    for col in probs:
        if "Leiden" in col or "Kmeans" in col:
            #print(obs[col],probs[col])
            obs[col+"_primaries"] = probs[col]

    df,obs,dfxy = clauto(df,obs,dfxy)
    obs = obs.loc[:,obs.columns.sort_values()]
    df,obs,dfxy = refinetype(df,obs,dfxy)
    df,obs,dfxy = aitype(df,obs,dfxy)
    df,obs,dfxy = aiSecondary(df,obs,dfxy)
    df,obs,dfxy = aiSecondary1(df,obs,dfxy)
    df,obs,dfxy = neighborhoodFractions(df, obs, dfxy)
    df,obs,dfxy = neighborhoodType(df,obs,dfxy)
    print(obs.columns)
    #print(df.columns)
    #obs = obs.drop('Primary Celltype SVM_phenotype',axis=1)  #WHYYYY
    df,obs,dfxy = receptorType(df,obs,dfxy)
    return([df,obs,dfxy])#,prdf,nedf])


def saveTable(dfs,keySs):   #looking for agreement in primary celltype?
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    colA = []
    for keyS in keySs:
        cols = []
        for col in obs.columns:
            if keyS in col:
                if len(col.split(keyS)[0]) == 0:
                    #print(col)
                    continue
                cols.append(col)
        colA.append(cols)
    #print(colA,"COLS!!!")
    for i in range(len(keySs)-1):
        odf= pd.DataFrame(0,columns=colA[i],index=colA[i+1])
        cat1,cat2 = keySs[i],keySs[i+1]
        for c1 in colA[i]:
            for uc in obs.loc[:,c1].unique():
                sobs = obs.loc[obs.loc[:,c1] == uc,:]
                for c2 in colA[i+1]:
                    tobs = sobs.loc[obs.loc[:,c2] == uc,:]
                    odf.loc[c2,c1] += tobs.shape[0]
        odf.to_csv(cat1+"_"+cat2+".csv")







def processing(dfs):
    toDrop = ["DAPI","R0c","R6Q","cytoplasm_n"]+BADMARKERl
    ocols = dfs[0].columns
    os = dfs[0].shape
    dfs = autoClean(dfs)
    dfs = dropCols(dfs,toDrop)
    dfs = simplePart(dfs)
    dfs[1]["slide_scene"] = ["_".join(ind.split("_")[:-1]) for ind in dfs[1].index]


    gs = ''
    for ss in dfs[1]["slide_scene"].unique(): #only scene 2
        if '2' in ss:
            gs = ss
    key = dfs[1]["slide_scene"] == gs
    print(key.sum())
    ndfs = []
    for DF in dfs:
        ndfs.append(DF.loc[key,:])
    dfs = ndfs

    dfs = reorder(dfs)
    #print(dfs[0].columns)
    OUT.append([os,dfs[0].shape,ocols,dfs[0].columns])
    return(dfs)



def save(data,foln,filn,typ="csv",labs=''):
    for i in range(100):
        print('save!')
    if ":" in filn:
        filn = filn.replace(":",".")
    if ":" in foln:
        foln = foln.replace(":",".")
    #if not checkL(foln,os.listdir(savedir)):
    if not os.path.isdir(savedir+"/"+foln):
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        os.mkdir(savedir+"/"+foln)
    if typ == "csv":
        data.to_csv(savedir+"/"+foln+"/"+filn+'.csv')
    if typ == "png":
        return(savedir+"/"+foln+"/"+filn+'.png')
    if typ == 'ome.tiff':
        #print(data.shape)
        svn = savedir+'/'+foln+'/'+filn+str(randint(0,1000))+'.ome.tiff'
        tifffile.imwrite(svn,data=data,metadata={"axes": "CYX","Labels": labs})

'''
'immune subtype autoCellType res: 1.0',
                                     'tumor subtype autoCellType res: 1.0',
                                     'cytotoxic autoCellType res: 1.0',
                                     'proliferating autoCellType res: 1.0',
                                     'receptors autoCellType res: 1.0',]
'''


def saveOmeTiff(obs,tran=False,keyS=['SVM_','high']):
    global SPI
    print("CURRENTLY NOT SAVING TIFF !!!")

    #'rolif',
    cols = []
    for c in obs.columns:
        for k in keyS:
            if k in c and c not in cols:
                cols.append(c)
    for ss in obs.loc[:,"slide_scene"].unique():
        segpath = segPaths[SPI]
        ss = str(ss)
        imAL,names = zapari(obs,cols,ss,segpath)
        SPI += 1
        imA=np.array(imAL)
        try:
            pass
            #save(imA,BX+'_ome.tiffs',ss,typ='ome.tiff',labs=names)  #SAVE .ome.tiff
        except:
            print("could not save ome tiff")
        for i in range(len(imAL)):
            nam = names[i]
            if tran:
                im0 = PIL.Image.fromarray(imAL[i])
                im = im0.convert("RGBA")
                datas = im.getdata()
                newData = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
                        # storing a transparent value when we find a black colour
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append((255, 255, 255))
                    #print(newData)
                im.putdata(newData)

            else:
                im = PIL.Image.fromarray(imAL[i])
            im.save(sphelper(savedir+"/"+BX+ss+'_PNGs',nam),"PNG")



def sphelper(foln,filn):
     #if not os.path.isdir(foln):
         #foln = os.getcwd()
     badchars = [":","/","?",">","<"]
     for bc in badchars:
         if bc in filn:
             filn = filn.replace(bc,".")
     if not os.path.isdir(foln):
         print(foln)
         if True:#input("folder does not exist. Try to create folder?") == 'y':
             os.mkdir(foln)
         else:
             savefold = input("path to folder:")
             return(sphelper(savefold,filn))
     return(foln+"/"+filn+'.png')




def zapari(obs,columns,slideScene,segpath,cellidn="cellid",maxEnts=20):
    print("running zapari...")
    OUTLINE = True #False#
    #blank = np.empty((), dtype=object)
    #blank[()] = (255,255,255,0)
    #VAL = np.empty((), dtype=object)
    #VAL[()] = (255,255,255)
    VAL = 25000   #at 1, worked for transparent but not nontransp
    keyS = obs.loc[:,"slide_scene"] == slideScene
    obs = obs.loc[keyS,:]
    obs = obs.astype(str)
    label = io.imread(segpath)
    iS = label.shape
    #print(label.shape)
    #ucells = np.unique(label)
    if OUTLINE:
        bounds = skimage.segmentation.find_boundaries(label, connectivity=1, background=0)
    imAL = []
    names = []
    for column in columns:
        print("...",column)
        obs["intcol"] = 0
        uEnts = sorted(list(obs.loc[:,column].unique()))#.astype(str)
        if len(uEnts) > maxEnts:
            continue
        for i,e in enumerate(uEnts):
            #print("cluster:",e)
            #chan = np.full_like(label,blank,dtype=object)
            chan = np.zeros_like(label)
            #ochan = np.zeros(iS)
            key = obs.loc[:,column] == e
            sobs = obs.loc[key,:]
            incl = list(sobs.loc[:,cellidn])
            #print(len(incl))
            for cell in incl:
                #print(i)
                l = int(cell.split("cell")[-1])

                if OUTLINE:
                    chan[np.logical_and(label==l, bounds)] = VAL
                else:
                    chan[label==l] = VAL
            imAL.append(np.copy(chan))
            names.append(column+" "+e)
    return(imAL,names)

'''
'''

def aitype(df,obs,dfxy,keyS = "refined"): #must have agreement on every category with keystring. swap to "rimary" to go without the refining step
    thres = .99
    cols = []
    for col in obs.columns:
        if keyS in col:
            cols.append(col)
    robs = obs.loc[:,cols]
    mtype = robs.mode(axis=1).iloc[:,0]
    mty = list(mtype)
    mtype = []
    while len(mtype) < robs.shape[1]:
        mtype.append(mty)
    mtype = np.array(mtype).T
    #print(mtype,'mtype\n')
    #print(mtype)
    key = np.equal(mtype,robs)  #.values
    #print(key,'key',"\n")
    mct = key.sum(axis=1)
    mct = mct/len(cols)
    #print(mct,"mct\n")
    tKey = mct > thres
    #print(tKey,"tKey\n")
    #print(robs.loc[~tKey,:],"below thresh")
    tdf,tobs,txy = df.loc[tKey,:],obs.loc[tKey,:],dfxy.loc[tKey,:]
    #print(df.shape,tdf.shape,"dataset shape, training set shape")
    X,Y = sv.buildData(tdf,tobs,txy,yind=cols[0])
    inn = "SAA16"+BX
    sv.trainSVM(X,Y,name=inn+"_SVM")
    sv.trainSVM(X,Y,name=inn+"_NN",mode="nn")
    X,Y = sv.buildData(df,obs,dfxy,yind=cols[0])
    obs["SVM_primary"],obs["NN_primary"] = "",""
    obs["SVM_primary"] = sv.useSVM(X,name=inn+"_SVM")
    obs["NN_primary"] = sv.useSVM(X,name=inn+"_NN")
    return(df,obs,dfxy)


def refinetype(df,obs,dfxy,colS = 'rimary',keyS = 'tromal'):
    rres = 1.5
    rncl = 20
    for col in obs.columns:
        if colS in col:
            obs[col+"_refined"] = obs.loc[:,col]
            uty = sorted(list(obs.loc[:,col].unique()))
            for ty in uty:
                if keyS in ty:
                    key = obs.loc[:,col]==ty
                    sdf,sobs,sxy = df.loc[key,:],obs.loc[key],dfxy.loc[key,:]
                    sdf,sobs,sxy = autoleiden(sdf,sobs,sxy,rncl)
                    sdf,sobs,sxy = clauto(sdf,sobs,sxy,NAM=[str(rncl)],res=rres)
                    for col1 in sobs.columns:

                        if "eiden" in col1 and "rimary" in col1 and str(rres) in col1:
                            print(col1,"!!!")
                            obs.loc[key,col+"_refined"] = sobs.loc[:,col1]


    return(df,obs,dfxy)


def clauto(df,obs,dfxy,NAM = ['means','eiden','dgram','GMM'], res =.75 ):
    obs = obs.astype(str)
    #print(obs.shape)
    oobs = obs.copy()

    chs, uchs = [],[]
    goodcols = []
    for col in obs.columns:
        for nam in NAM:
            if nam in col and col not in goodcols:
                goodcols.append(col)
    for col in goodcols:
        chs.append(list(obs.columns).index(col))
        uchs.append(obs.loc[:,col].unique())
    for i,ch in enumerate(chs):
        uch = uchs[i]
        adf,aobs,axy = clag(df,obs,dfxy,ch,uch)
        x,aobs,xx = autotype(adf,aobs,axy,chanT=False,name=obs.columns[ch]+"cluster autotype",res=res)
        #print(obs.shape)
        for col in aobs.columns:
            if obs.columns[ch]+"cluster autotype" in col:
                #print(col,aobs.loc[:,col].unique(),"!!")
                obs[col] = ""
                for uc in aobs.index:
                    key = obs.iloc[:,ch] == uc
                    obs.loc[key,col] = aobs.loc[uc,col]
        #print(obs.shape,df.shape)
        #print(obs,df)
    return(df,obs,dfxy)




def clag(df,obs,dfxy,ch=None,uch=None,z=True):

    #if not ch:
        #ch,uch=obMenu(obs,"obs category to auto-annotate cell types")
    if z:
        zdf,zobs,zxy = zscorev(df,obs,dfxy)
    else:
        zdf,zobs,zxy = df,obs,dfxy
    ocol = obs.columns[ch]
    ndf,nobs,nxy = [],[],[]
    for uc in uch:
        key = zobs.loc[:,ocol] == uc
        #if key.sum() == 0:
            #nobs.append(pd.DataFrame(data=np.full_like(zobs,"no mode")).iloc[0,:])
            #continue
        sdf = zdf.loc[key,:]
        sobs = zobs.loc[key,:]
        sxy = zxy.loc[key,:]
        ndf.append(sdf.mean(axis=0))
        nxy.append(sxy.mean(axis=0))
        #print(sobs.mode(axis=0).iloc[0,:],"/n/n")
        #time.sleep(1)
        #print(sobs,sobs.shape)
        smo = sobs.mode(axis=0)
        #print(smo,"smo")
        nobs.append(smo.iloc[0,:])
    dfs = [ndf,nobs,nxy]
    for i,d in enumerate(dfs):
        dfs[i] =pd.concat(d,axis=1).transpose()
        dfs[i].index = uch.astype(str)
        #print(dfs[i].columns)
        #print(dfs[i].shape)
        #print(dfs[i])
    return(dfs[0],dfs[1],dfs[2])


def reorder(dfs,primaries=False): #also used for selcting clustering markers
    df = dfs[0]
    bioms = ['panCK', 'Ecad', 'CK7', 'CK8', 'CK19', 'CK5', 'CK14', 'CK17', 'MUC1', 'CD44', 'AR', 'ER', 'PgR', 'HER2', 'EGFR', 'GATA3', 'CoxIV', 'pS6RP', 'H3K4',
             'H3K27', 'pHH3', 'Ki67', 'PCNA', 'pRB', 'CCND1', 'CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'CSF1R', 'PD1', 'FoxP3', 'GRNZB', 'PDPN', 'PDL1', 'CD31',
             'CAV1', 'ColIV', 'ColI', 'aSMA', 'Vim', 'CD90', 'pAKT', 'pERK', 'pMYC', 'ZEB1', 'BMP2', 'TUBB3', 'Glut1', 'Pin1', 'gH2AX', 'Rad51', 'pRPA', 'TP53', '53BP1',
             'CC3', 'cPARP', 'BCL2', 'MHS2', 'MHS6', 'LaminAC', 'LamB1', 'LamB2']
    if primaries:
        bioms = ['panCK', 'Ecad', 'CK7', 'CK8', 'CK19', 'CK5', 'CK14', 'CK17', 'MUC1', 'AR', 'ER', 'PgR', 'HER2', 'EGFR',
                 'pHH3', 'Ki67', 'PCNA', 'CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'PD1', 'FoxP3', 'CD31',
                 'CAV1', 'ColI', 'aSMA', 'Vim', 'CD90']
    ocols = []
    for b in bioms:
        for col in df.columns:
            if b+"_" in col:
                ocols.append(col)
    if not primaries:
        for col in df.columns:
            if col not in ocols:
                ocols.append(col)
    dfs[0] = df.loc[:,ocols]
    #print(ocols)
    #print(dfs[0].columns,"df cols in reorder")
    return(dfs)



def onlyPrimaries1(df,obs,dfxy):
    prim = []
    for biom in df.columns:
        if fillMap(biom) != None:
            prim.append(biom)
        elif "area" in biom or "eccentr" in biom or "Ki67" in biom or "PCNA" in biom:
            prim.append(biom)
    df = df.loc[:,sorted(prim)]
    return(df,obs,dfxy)




def simplePart(dfs):
    df = dfs[0]
    ocols = []
    bioms = []
    for col in df.columns:
        if "nuclei_" in col:
            ocols.append(df.loc[:,col])
            continue
        cn = col.split("_")[0]+"_"
        if cn not in bioms:
            bioms.append(cn)
    for bn in bioms:
        serL = []
        for col in df.columns:
            if bn in col:
                serL.append(df.loc[:,col])
        if len(serL) > 1:
            bdf = pd.concat(serL,axis=1)
            outS = bdf.max(axis=1)
        else:
            outS = serL[0]
        outS.name = bn
        ocols.append(outS)
    odf = pd.concat(ocols,axis=1)
    dfs[0] = odf
    return(dfs)


def dropCols(dfs,strs):
    lis = [dfs[0]]
    for k,d in enumerate(lis):
        #print(list(d.columns))
        toRem = strs
        #print("\nbefore:\n",lis[k].columns)
        dr = []
        for col in d.columns:
            for t in toRem:
                if t in col:
                    dr.append(col)
        lis[k] = tryDrop(d,dr)
        #print("after:\n",lis[k].columns)
        #for d in lis:
        #print("\n",d.columns)
    return([lis[0],dfs[1],dfs[2]])

def tryDrop(df,dropList):
    for colName in dropList:
        try:
            df = df.drop([colName],axis = 1)
        except:
            #pass
            print(colName,'not in dataframe')
    return(df)

def autoClean(dfs):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
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
    return([df,obs,dfxy])



def checkL(S,L):
    for ent in L:
        if ent in S:
            return(True)
    return(False)






'''
#visu
'''


def silh(df,obs,dfxy):
    #print(obs.columns,'into silh')
    obs = obs.fillna("nan")
    cols = []
    scores = []
    for col in sorted(obs.columns):
        if "rimary" not in col:
            continue
        ncat = len(obs.loc[:,col].unique())
        if ncat < 2 or ncat > 20:
            continue
        try:
            sco = silhouette_score(df,obs.loc[:,col])
            silhouette_vals = silhouette_samples(df,obs.loc[:,col])
            #silhH1(silhouette_vals,obs.loc[:,col])
            #print(col,sco)
            cols.append(col)
            scores.append(sco)
        except Exception as e:
            print("could not score",col,"    ",e)
    plt.scatter(cols,scores)
    plt.xticks(rotation = 90)
    plt.grid(visible=True,axis='x')
    if SAVEfIGS:
        plt.savefig(save(0,BX+"_silhouette",'scatter',typ="png"),bbox_inches='tight')
    plt.show()
    return(df,obs,dfxy)

def silhH1(silhouette_vals,labels):
    fig,ax1 = plt.subplots()
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    plt.show()

def getcombs(tcols):
    allcombs = []
    listLens = 1
    for tl in tcols:
        listLens *= len(tl)
    for i in range(listLens):
        allcombs.append([])
    for tl in tcols:
        ll = len(tl)
        i = 0
        while i < listLens:
            allcombs[i].append(tl[i%ll])
            i += 1
    return(allcombs)

def callMap(df,obs,dfxy,slide=False):
    dfs = [df,obs,dfxy]
    if slide:
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],['slide'],['proliferating autoCellType res: 1.0']])
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[['slide'],["SVM_primary"],["tumor subtype SVM_phenotype"],["immune subtype SVM_phenotype"]])#,keyS = "sub")# "autoCell"
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],['2 immune_high'],['1 endothelial_high'],['slide']])
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],['slide']])

        return()
    else:
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],["tumor subtype SVM_phenotype"],["immune subtype SVM_phenotype"],['proliferating autoCellType res: 1.0']])#,keyS = "sub")# "autoCell"
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],['Leiden_22_tumor'], ['Leiden_22_immune'],['Leiden_22_other']])
        sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["SVM_primary"],['2 immune_high'],['1 endothelial_high']])
    #sortedMap(dfs[0],dfs[1],dfs[2],tcols=[["tumor subtype SVM_phenotype","immune subtype SVM_phenotype"],['proliferating autoCellType res: 1.0']])
    for i in range(3):
        nams  = ["tumor","immune","other"]
        nam = nams[i]
        if i == 0:
            key = obs.loc[:,'SVM_primary'] == "3 tumor"
            tcols = [["SVM_primary"],["tumor subtype SVM_phenotype"],['receptors SVM_phenotype'],['proliferating autoCellType res: 1.0'],['SVML_phenotype']]
        elif i == 1:
            key = obs.loc[:,'SVM_primary'] == "2 immune"
            tcols = [["SVM_primary"],["immune subtype SVM_phenotype"],['cytotoxic SVM_phenotype'],['immune checkpoints SVM_phenotype'],['proliferating autoCellType res: 1.0'],['SVML_phenotype']]
        else:
            k1 = obs.loc[:,'SVM_primary'] != "3 tumor"
            k2 = obs.loc[:,'SVM_primary'] != "2 immune"
            key =  k1 & k2
            tcols = [["SVM_primary"],['proliferating autoCellType res: 1.0'],['SVML_phenotype']]

        sdf,sobs,sxy = df.loc[key,:],obs.loc[key,:],dfxy.loc[key,:]
        sortedMap(sdf,sobs,sxy,tcols = tcols,dgram=True)
        #singHeatmap(sdf,sobs,sxy,tcols=tcols)
        ch = list(sobs.columns).index('Leiden_22_'+nam)
        uch = sobs.loc[:,'Leiden_22_'+nam].unique()
        print(sobs.shape,"sobs shape")
        sdf,sobs,sxy = clag(sdf,sobs,sxy,ch=ch,uch=uch)
        tcols += [['2 immune_high'],['1 endothelial_high']]
        sortedMap(sdf,sobs,sxy,tcols = tcols,figtitle=nam,dgram=True)


def sortedMap(df,obs,dfxy,tcols = [["SVM_primary"],["immune subtype SVM_phenotype","tumor subtype SVM_phenotype"],
                                  ['proliferating autoCellType res: 1.0']],
              figtitle = '',dropNA = True,dgram= False, sfs = False):
    #make sort-by-dendrogram function that adds "1- 2-" in front of any label so this has dendrogram option
    #then can run aggregate by leiden and run through here (after sorting highest level label (or any) by dendrogram)

    obs = obs.astype(str)
    odf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()

    mpl.style.use('default')
    sns.set(font_scale=2)


    allcombs = getcombs(tcols)
    print(allcombs,"allcombs!")
    for cols in allcombs:
        sns.set(font_scale=2)
        col = cols[0]
        if dropNA:
            nkey = oobs[col].isna()
            nk2 = oobs[col] == "nan"
            nkey = nkey | nk2
            df,obs,dfxy = odf.loc[~nkey,:],oobs.loc[~nkey,:],oxy.loc[~nkey,:]
            df,obs,dfxy = zscorev(df,obs,dfxy)
        allutys = []
        for col in cols:
            utys = sorted(list(obs.loc[:,col].astype(str).unique()))
            allutys.append(utys)
        data,colors = sortMap(df,obs,dfxy,cols,allutys)
        #print("\n\n",colors)
        vout = 10
        ax=sns.clustermap(data, vmin=-vout, vmax=vout, cmap='bwr',row_colors=colors,
                              yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                              row_cluster=False, col_cluster=False, colors_ratio = 0.01)

        if SAVEfIGS or sfs:
            plt.savefig(save(0,BX+"_single-heatmap",figtitle+" "+"_".join(cols),typ="png"),bbox_inches='tight')
        plt.show()
        if dgram != False:
            singHeatmap(df,obs,dfxy,tcols=tcols,title=figtitle+" d-gram")
        mpl.style.use('default')
        fig = plt.figure(figsize=(4*len(cols),8))
        primts = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","5 stromal"]
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
                elif uo in primts:
                    uoi = primts.index(uo)
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
            ax.set_xticklabels(kl,rotation=90,size=min(10,80/len(utys)))
            #ax.title.set_text(col,rotation = 90)
            ax.set_ylabel(col)
            #ax.set_title(col)
            ax.set_yticks([])
            plt.tight_layout()
        if SAVEfIGS or sfs:
            plt.savefig(save(0,BX+"_single-heatmap",figtitle+" legend_"+"_".join(cols),typ="png"),bbox_inches='tight')
        plt.show()






def sortMap(df,obs,dfxy=9,cols=[["no cols included"]],allutys = None):
    data,colors = [],[]
    while len(colors) < len(cols):
        colors.append([])
    col = cols[0]
    utys = allutys[0]
    primts = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","5 stromal"]
    for i,uo in enumerate(utys):
        key = obs.loc[:,col]==uo
        #print(key.sum())
        if key.sum() == 0:
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
            data.append(d1)

            for j,cS in enumerate(c1):
                colors[j+1].append(cS)
        else:
            data.append(sdf)
    try:
        data = pd.concat(data,axis=0)
    except:
        print(utys,"no vals to concat in data",obs.loc[:,col])
    c = []
    for ci in range(len(colors)):
        cs = pd.concat(colors[ci],axis=0,ignore_index=False)
        c.append(cs)
    colors = c
    return(data,colors)



def singHeatmap(df,obs,dfxy,vout=10,ress=[[5,2.5]],tcols = ["SVM_primary","immune subtype SVM_phenotype","tumor subtype SVM_phenotype",
                                  'proliferating autoCellType res: 1.0'],title=""):
    sns.set(font_scale=2)
    df,obs,dfxy = zscorev(df,obs,dfxy)

    colCol = obs.iloc[:,0].copy()
    colCol[:] = "white"
    cols = []
    primts = ["1 endothelial","2 immune","3 tumor","4 active fibroblast","5 stromal"]
    kws = []
    for col in tcols:
        if type(col) == list:
            col = col[0]
        colCol.name = col
        uty = sorted(list(obs.loc[:,col].unique()))
        for i,ut in enumerate(uty):
            key = obs.loc[:,col] == ut
            if ut == "nan" or ut == "":
                colCol.loc[key] = "lightgray"
            elif ut in primts:
                uoi = primts.index(ut)
                colCol.loc[key] = allc.colors[uoi]
            else:
                colCol.loc[key] = allc.colors[i]

        cols.append(colCol.copy())
        kws.append(col)
    stt = time.time()
    #fy,ay = plt.subplots()
    ytl = True
    if df.shape[0] > 50:
        ytl = False
    ay = sns.clustermap(df,yticklabels=ytl, xticklabels=True,center=0, cmap='bwr',
                        figsize=(25,25),vmin=-vout, vmax=vout, row_colors = cols,
                         colors_ratio = 0.01) #cbar_kws = {'label':kws},
    #f1 = fy.number
    #plt.show()
    dgram = ay.dendrogram_row.dendrogram
    #print(dgram)
    #sortD = df.iloc[dgram["leaves"],:]
    #rpn = 0
    for rp in ress:
        res,sres =rp[0],rp[1]
        distances = []
        for i in range(df.shape[0]-1):
            c1 = df.iloc[dgram["leaves"][i],:]
            c2 = df.iloc[dgram["leaves"][i+1],:]
            dists = (c1-c2)**2
            #print(dists)
            ds = dists.sum()**.5
            distances.append(ds)
        #fz,az = plt.subplots()
        #if rpn == 0:
        #    az.plot(distances,np.ones(len(distances))*len(distances)-1-np.arange(len(distances)),marker="o",markersize=.5,linestyle="None")
        #rpn += 1
        md = stat.mean(distances)
        sd = stat.stdev(distances)
        lines = []

        i = 0
        lastD = None
        dc = 0
        cN = "dgram cluster_"+str(res)+"x"+str(sres)+title
        obs[cN] = 9999
        while i < len(distances):
            #print(i)
            d = distances[i]
            if d > md+sd*sres:
                nearCells = distances[max(0,i-res):min(i+res,len(distances))]
                m = distances.index(max(nearCells))
                #print(i,m)
                if i == m:
                    lines.append(m)
                    if lastD != None:
                        lev = dgram['leaves'][lastD+1:i+1]
                        #print(lev)
                        obs.loc[:,cN].iloc[lev] = dc
                        dc += 1
                    lastD = i
                    i += res+1
                elif m > i:
                    i = m
                else:
                    i += 1
            else:
                i += 1
        obs.loc[obs.loc[:,cN]== 9999,cN] = dc
        #fx,ax=plt.subplots()
        ax = ay.ax_heatmap
        #print(lines)
        for l in lines:
            ax.plot([0,df.shape[1]],[l+1,l+1], 'k-', lw = 1)
        fig=ax.get_figure()
        if SAVEfIGS:
            fig.savefig(save(0,BX+"_single-heatmap",cN,typ="png"),bbox_inches='tight')
        plt.show()

        print((time.time()-stt)/60," minute runtime singlecell clustermap")

    mpl.style.use('default')
    return(obs)

def singHeatmap1(df,obs,dfxy,pc=True,vout=10,ress= [[50,1],[25,2]]):
    mpl.style.use('default')
    #res = 50 #must be highest distance for this number of distances each direction
    #sres = 1
    sns.set(font_scale=2)
    df,obs,dfxy = zscorev(df,obs,dfxy)
    #for i,col in enumerate(obs.columns):
        #print(i,col)
    cols = []
    ctainp = [2, ]#5z, 7, 9, 6]
    cochinp = [['blue','red','yellow'],
               ['#BDBDBD', '#32CD32', '#0000FF', '#FFA000', '#FF0000',],
               ['#BDBDBD', 'black', 'red', 'blue','green','orange'],
               ['#BDBDBD', 'red', 'blue', 'yellow', 'black',],
               ['#BDBDBD', 'red']]
    for i in range(len(ctainp)):
        cta = ctainp[i]
        cochl = cochinp[i]
        colCol = obs.iloc[:,cta].copy()

        colCol[:] = "gray"
        uty = obs.iloc[:,cta].unique()
        for j in range(len(uty)):
            try:
                colCol.loc[obs.iloc[:,cta]==uty[j]] = cochl[j]
            except:
                colCol.loc[obs.iloc[:,cta]==uty[j]] = "white"
                print("out of colors, using white",obs.columns[cta])
        cols.append(colCol)


    stt = time.time()
    #fy,ay = plt.subplots()
    ay = sns.clustermap(df,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(25,25),vmin=-vout, vmax=vout, row_colors =cols)#
    #f1 = fy.number
    #plt.show()
    dgram = ay.dendrogram_row.dendrogram
    #print(dgram)
    #sortD = df.iloc[dgram["leaves"],:]
    rpn = 0
    for rp in ress:
        res,sres =rp[0],rp[1]
        distances = []
        for i in range(df.shape[0]-1):
            c1 = df.iloc[dgram["leaves"][i],:]
            c2 = df.iloc[dgram["leaves"][i+1],:]
            dists = (c1-c2)**2
            #print(dists)
            ds = dists.sum()**.5
            distances.append(ds)
        fz,az = plt.subplots()
        if rpn == 0:
            az.plot(distances,np.ones(len(distances))*len(distances)-1-np.arange(len(distances)),marker="o",markersize=.5,linestyle="None")
        rpn += 1
        md = stat.mean(distances)
        sd = stat.stdev(distances)
        lines = []

        i = 0
        lastD = None
        dc = 0
        cN = "dgram cluster_"+str(res)+"x"+str(sres)
        obs[cN] = 9999
        while i < len(distances):
            #print(i)
            d = distances[i]
            if d > md+sd*sres:
                nearCells = distances[max(0,i-res):min(i+res,len(distances))]
                m = distances.index(max(nearCells))
                #print(i,m)
                if i == m:
                    lines.append(m)
                    if lastD != None:
                        lev = dgram['leaves'][lastD+1:i+1]
                        #print(lev)
                        obs.loc[:,cN].iloc[lev] = dc
                        dc += 1
                    lastD = i
                    i += res+1
                elif m > i:
                    i = m
                else:
                    i += 1
            else:
                i += 1
        obs.loc[obs.loc[:,cN]== 9999,cN] = dc
        #fx,ax=plt.subplots()
        ax = ay.ax_heatmap
        #print(lines)
        for l in lines:
            ax.plot([0,df.shape[1]],[l+1,l+1], 'k-', lw = 1)
        fig=ax.get_figure()
        if SAVEfIGS:
            fig.savefig(save(0,BX+"_single-heatmap",cN,typ="png"),bbox_inches='tight')
        plt.show()

        print((time.time()-stt)/60," minute runtime singlecell clustermap")
    mpl.style.use('default')
    return(obs)



def clusterbar(cdf,obs,dfxy):
    #print(obs.shape,"oshape in clusterbar")
    fig,ax = plt.subplots(figsize=(15,20))
        #rowColors = pd.DataFrame(sorted(list(obs["Cluster"].unique())))[0].map(rowColors)
    try:
        sns.set(font_scale=2)
        ay = sns.clustermap(cdf,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(25,20))#,metric="euclidean")
    except Exception as e:
        print(e,"error")
        return()
    plt.setp(ay.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    ayax = ay.ax_heatmap.yaxis.get_majorticklabels()
    #print(obs.shape,"oshape in clusterbar1")
    #for i,op in enumerate(obs.columns):
        #print(i,op)
    color = 'Primary Celltype autoCellType res: 1.0'
    ucols = obs[color].unique()
    a = np.zeros((len(ayax),len(ucols)))
    bins = []
    for i,Text in enumerate(ayax):
        Text=Text.get_text()
        key1 = obs["Cluster"].astype(str) == str(Text)
        #print(str(Text))
        bins.append(str(Text))
        #print(key1,"1")
        for j,c in enumerate(ucols):
            key2 = obs[color] == c
            #print(key2,"2")
            bi = obs.loc[key1 & key2,:]
            a[i,j] = bi.shape[0]
    fig, ax = plt.subplots(figsize=(10,20))
    shape = a.shape
    height = np.zeros(shape[0])
    #print(obs.shape,"oshape in clusterbar2")
    for i in range(shape[1]):
        try:
            ax.barh(bins,a[:,i],label=ucols[i],left=height,color=allc.colors[i])
        except:
            ax.barh(bins,a[:,i],label=ucols[i],left=height,color=allc.rgb())
        height += a[:,i]
    plt.gca().invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=color)
    #ax.title.set_text(" ".join(PROCESS))
    plt.show()
    mpl.style.use('default')
    #print(obs.shape,"oshape in clusterbar3")
    return()



'''
'immune subtype autoCellType res: 1.0',
                                     'tumor subtype autoCellType res: 1.0',
                                     'cytotoxic autoCellType res: 1.0',
                                     'proliferating autoCellType res: 1.0',
                                     'receptors autoCellType res: 1.0',
'''
def spatialLite(nobs,nxy,ymin=0,colNames = ['SVM_primary','proliferating SVM_phenotype', 'tumor subtype SVM_phenotype',
                                            'receptors SVM_phenotype', 'immune subtype SVM_phenotype',
                                            'immune checkpoints SVM_phenotype', 'cytotoxic SVM_phenotype',
                                            'SVML_phenotype', 'Primary Celltype SVML_phenotype',
                                            'proliferating SVML_phenotype', 'tumor subtype SVML_phenotype',
                                            'receptors SVML_phenotype', 'immune subtype SVML_phenotype',
                                            'immune checkpoints SVML_phenotype', 'cytotoxic SVML_phenotype',
                                            '2 immune_high', '1 endothelial_high','Primary Celltype autoCellType res: 1.0'],sfs = False):
    #['Primary Celltype autoCellType res: 1.0','Primary Celltype Leiden_30_primariescluster autotype0.75']

    nobs = nobs.astype(str)
    for scene in nobs["slide_scene"].unique():
        for CN in colNames:
            #colors = ["#FF0000","#32CD32","#0000FF","#FFA000","#BDBDBD"] # = allc.amcolors
            #uch = ['1 endothelial','2 immune','3 tumor','4 active fibroblast','5 stromal']
            colors = allc.colors
            uch = sorted(list(nobs.loc[:,CN].unique()))
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
                if ty == "" or ty == "nan" or ty == "no":
                    co = "lightgray"
                elif ty == "yes":
                    co = "darkred"
                #print(sobs.columns[ch1])
                key1 = sobs[CN]==ty
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
            plt.title(scene+" "+CN)
            if SAVEfIGS or sfs:
                print("SAVING !!! $$$$")
                fig=ax.get_figure()
                path = save(0,BX+"_spatial",scene+CN,typ="png")
                print(path,'path!')
                fig.savefig(path,bbox_inches='tight')
            plt.show()

def trimExtremes(ser,quantile=.97):
    sS = ser.sort_values()
    nInd = sS.shape[0]
    cutoff = sS.iloc[int(nInd * quantile)]
    #print(cutoff,"!")
    key = ser < cutoff
    ser = ser.loc[key]
    return(ser)

def hist(df,obs,dfxy,colNames= ['SVM_primary']): #['Primary Celltype autoCellType res: 1.0','Primary Celltype Leiden_30_primariescluster autotype0.75']
    for col in df.columns:
        for CN in colNames:
        #continue                                  # f!!!!!!!!!! !!! !!!!!!!!!!!!!!!!
            xs,ys = [],[]
            sCN = sorted(list(obs[CN].unique()))
            for ty in sCN:
                if "n_50" in col:
                    tdf = df.loc[obs[CN]==ty,:]
                    tcol = trimExtremes(tdf[col])
                    x,y = makeHist(tcol,tdf.shape[0]/2)
                    xs.append(x)
                    ys.append(y)
                else:
                    try:
                        tdf = df.loc[obs[CN]==ty,:]
                        tcol = trimExtremes(tdf[col])
                        x,y = makeHist(tcol,tdf.shape[0]/2)
                        xs.append(x)
                        ys.append(rollingAve(rollingAve(y,n=int(tdf.shape[0]/20)),n=int(tdf.shape[0]/50)))
                    except:
                        pass
            fig,ax=plt.subplots()
            for i in range(len(xs)):
                try:
                    x,y,c,ty = xs[i],ys[i],allc.amcolors[i],sCN[i]
                    ax.plot(x,y,c,label=ty, alpha=0.5)
                except:
                    print("hist can't use amcolors")
                    x,y,c,ty = xs[i],ys[i],allc.colors[i],sCN[i]
                    ax.plot(x,y,c,label=ty, alpha=0.5)
            ax.legend()
            ax.set_title(col+" "+CN)
            plt.ylabel("log2 cell counts")
            if SAVEfIGS:
                fig = ax.get_figure()
                fig.savefig(save(0,BX+"_histograms",col+CN,typ="png"),bbox_inches='tight')
            plt.show()

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


def boxplot1(df,obs,dfxy,title="all cells"): #shows several markers
    cats = ["slide"] #,"Primary Celltype autoCellType res: 1.0","dgram cluster_50x1"
    markers=[['CK7', 'CK8', 'CK19', 'CK5', 'CK14', 'CK17', 'Ecad', 'EGFR', 'MUC1', 'CD44'],
     ['ER', 'PgR', 'AR', 'HER2', 'GATA3', 'CoxIV', 'pS6RP', 'H3K4', 'H3K27'],
     ['Ki67', 'PCNA', 'pHH3', 'pRB', 'CCND1', 'BCL2'],
     ['CD45', 'CD3_', 'CD4_', 'CD8', 'CD20', 'CD68', 'PD1', 'FoxP3', 'GRNZB', 'PDPN', 'CSF1R'],
     ['CD31', 'CAV1', 'ColIV', 'aSMA', 'Vim', 'ColI', 'CD90', 'PDPN', 'CD44', 'Pin1'],
     ['pAKT', 'pERK', 'pMYC', 'ZEB1', 'BMP2', 'TUBB3', 'Glut1'],
     ['gH2AX', 'Rad51', 'pRPA53BP1', 'CC3', 'cPARP', 'MSH2', 'PDL1'],
     ['panCK', 'LamAC', 'LamB1', 'Pin1']]
    for cat in cats:
        try:
            if len(list(obs.loc[:,cat].unique())) < 2:
                continue
        except:
            print(cat,"missing")
            continue
        for ml in markers:
            toShow = []
            for col in df.columns:
                for m in ml:
                    if m in col:
                        toShow.append(col)
            if len(toShow) < 2:
                continue
            sdf = df.loc[:,toShow]
            #sdf.boxplot().plot()



            for us in obs.loc[:,cat].unique():
                tdf = sdf.loc[obs[cat]==us,:]
                mdf = pd.melt(tdf)
                fig,ax = plt.subplots(figsize=(8,8))
                try:
                    ax=sns.boxplot(data=mdf, x='variable',y='value',showfliers=False)
                except Exception as e:
                    print("cound not plot",us,e)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=us+"_"+cat)
                plt.xticks(rotation = 85)
                if SAVEfIGS:
                    try:
                        fig=ax.get_figure()
                        fig.savefig(save(0,BX+"_boxplots",us+"_"+cat+"_"+ml[0],typ="png"),bbox_inches='tight')
                    except:
                        print("could not save boxplot, too big")
                plt.show()

    return(df,obs,dfxy)


def callBox(df,obs,dfxy,bins = ['SVM_primary']):
    #"autoCellType res: 1.0",
    for b in bins:
        boxplot(df,obs,dfxy,b)

def boxplot(df,obs,dfxy,binCol):
    bins = sorted(list(obs.loc[:,binCol].unique()))
    #colorL = allc.colors[:len(bins)]
    #print(colorL)
    colorL = None
    if len(bins) < 2:
        return(df,obs,dfxy)
    colCol = None
    ucols = [1]
    dfo = df.merge(obs,left_index=True,right_index=True).sort_values(binCol)
    #print(dfo)
    for marker in df.columns:
        fig,ax = plt.subplots(figsize=(max(8,int(len(bins)*len(ucols)**.7/2)),8))
        ax=sns.boxplot(hue=colorL, data=dfo, x=binCol, y=marker,showfliers=False) #, hue_order=bins, order=bins
        try:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
        except:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation = 85)
        for j,obj in enumerate(list(ax.patches)):
            ax.patches[j].set_facecolor(allc.colors[j])



        if SAVEfIGS:
            fig=ax.get_figure()
            fig.savefig(save(0,BX+"_boxplots",marker+"_"+binCol,typ="png"),bbox_inches='tight')
        plt.show()
    return(df,obs,dfxy)

def showUmap(df,obs,dfxy,cdf=None,keySs = ['SVM_','autoCell',"_high"]): #make it so there's an option to re-color without re-genning whole map
    oobs = obs.copy()
    #26obs = obs.astype(str)
    for com in obs.columns:
        utys = obs.loc[:,com].unique()
        for ut in utys:
            if ut == "no" or ut == "nan":
                key = obs.loc[:,com] == ut
                obs.loc[key,com] = np.nan
    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.umap(adata)
    plt.rcParams['figure.figsize'] = 8, 8
    #comL = ['slide','Primary Celltype autoCellType res: 1.0','proliferating autoCellType res: 1.0','tumor subtype autoCellType res: 1.0',
    #        'receptors autoCellType res: 1.0','immune subtype autoCellType res: 1.0','immune checkpoints autoCellType res: 1.0',
    #        'cytotoxic autoCellType res: 1.0','fibroblast type autoCellType res: 1.0',"celltype_bx"]
    comL = []
    for col in obs.columns:
        for keyS in keySs:
            if keyS in col and col not in comL:
                if len(list(obs.loc[:,col].unique())) > 10:
                    continue
                comL.append(col)

    for com in comL:
        utys = obs.loc[:,com].unique()
        if len(utys) == 1:
            continue
        try:
            if "rimary" in com:
                tys = obs.loc[:,com].unique()
                palette = dict(zip(sorted(list(tys)),allc.colors[:len(tys)]))
                print(palette)
                ax=sc.pl.umap(adata,color = com,na_color='lightgray',show=False,palette=palette)
            else:
                ax=sc.pl.umap(adata,color = com,na_color='lightgray',show=False)
        except:
            print("could not make umap for",com)
            continue
        if SAVEfIGS:
            fig=ax.get_figure()
            fig.savefig(save(0,BX+"_umaps",com,typ="png"),bbox_inches='tight')
        plt.show()
    for biom in df.columns:
        ay=sc.pl.umap(adata,color=biom,vmin=np.mean(df[biom])-np.std(df[biom]),
                      vmax=np.mean(df[biom])+2*np.std(df[biom]),color_map='viridis',show=False)
        if SAVEfIGS:
            fig=ay.get_figure()
            fig.savefig(save(0,BX+"_umaps",biom,typ="png"),bbox_inches='tight')
        else:
            plt.show()
    return(df,oobs,dfxy)

def callBar(df,obs,dfxy):
    #print('need to update barplots!')
    '''
    sepL = ['Primary Celltype autoCellType res: 1.0',
            'Kmeans 7_primaries', 'Leiden_0.3_primaries']
    comL = ['tumor subtype autoCellType res: 1.0',
            'receptors autoCellType res: 1.0','immune subtype autoCellType res: 1.0',
            'immune checkpoints autoCellType res: 1.0',
            'cytotoxic autoCellType res: 1.0','proliferating autoCellType res: 1.0','Primary Celltype autoCellType res: 1.0']
    '''
    sepL = []

    for col in obs.columns:
        if 'SVM_pr' in col or "subtype SVM_" in col:
            sepL.append(col)
    badS = ["SVML","Leiden","ing SVM_phe","ors a","xic a","nts a"]
    for sep in sepL:
        keys = ["prolif","high"]
        if "tum" in sep:
            keys+= ["receptors"]
        elif "imm" in sep:
            keys += ["check","cytot"]
        for com in obs.columns:
            switch = 0
            for ks in keys:
                if ks in com:
                    for bs in badS:
                        if bs in com:
                            switch = 1
                    if switch == 0:
                        if len(list(obs.loc[:,com].unique())) < 10 and len(list(obs.loc[:,com].unique())) < 20:
                            barplot(df,obs,dfxy,sep,com)
                            break
    #'''

def barplot(df,obs,dfxy,binCol,colCol,showNA=True,piechart = False,biomchart=True):
    if not showNA:
        #print(obs.shape)
        obs = obs.loc[~pd.isna(obs[colCol]),:]
        #print(obs.shape,"!!")
    oobs = obs.copy()
    obs = obs.astype(str)
    #rch = int(input("show actual number (0) or percentages (1)?"))
    #for i,he in enumerate(obs.columns):
        #print(i,he)
    #ch = int(input("sort x axis by:"))
    obs = obs.sort_values(binCol,axis=0) #THIS MESSES UP THE INDEX
    bins = list(obs.loc[:,binCol].unique())

    #for i,he in enumerate(obs.columns):
        #print(i,he)
    #ch2 = int(input("color bars by:"))
    colors = sorted(list(obs.loc[:,colCol].unique()))
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
            c[i,j] = n/nClrs[j]
    ctots = np.sum(c,axis=1)
    #print(ctots,ctots.shape)
    for i in range(c.shape[0]):
        d[i,:] = c[i,:]/ctots[i]*100
    #print(bins,len(bins),len(bins[0]))
    if biomchart:
        in2 = True
        in3 = False
        #if input("use z-scored (by biomarker) data? (y) for errorbarplot") == "y":
        #    in2 = True
        #if input("pick colors for errorbarplot? (y)") == "y":
        #    in3 = True
        zdf,ttt,tttt = zscorev(df,obs,dfxy)
        #barx = 0
        for binName in bins:
            kEY1 = obs.loc[:,binCol] == binName
            if kEY1.sum() < 2:
                continue
            for coloName in sorted(list(obs.loc[:,colCol].unique())):
                kEY2 = obs.loc[:,colCol] == coloName
                if in2:
                    bdf = zdf.loc[kEY1 & kEY2,:]
                else:
                    bdf = df.loc[kEY1 & kEY2,:]
                if bdf.shape[0] > 2:
                    fig1,ax1=plt.subplots(figsize=(bdf.shape[1]/5,6))
                    mean = bdf.mean(axis=0)
                    stdev = bdf.std(axis=0)
                    if in3:
                        keyss = []
                        colorss = []
                        while True:
                            keyS = input("key string for color assignment (sent blank to exit)")
                            if keyS == "":
                                break
                            else:
                                keyss.append(keyS)
                                colorss.append(input("color for bars containing this string"))
                        for i in range(bdf.shape[1]):
                            cn = bdf.columns[i]
                            for j,kS in enumerate(keyss):
                                if kS in cn:
                                    clr = colorss[j]
                                    ax1.errorbar(i,mean[i],yerr=stdev[i],
                                                 marker="o",ls='none',ecolor=clr,c="black")
                                    break
                            else:
                                ax1.errorbar(i,mean[i],yerr=stdev[i],
                                             marker="o",ls='none',ecolor="black",c="black")




                    else:
                        ax1.errorbar(np.arange(mean.shape[0]),mean,yerr=stdev,
                                     marker="o",ls='none')
                    ax1.set(title=binCol+":"+binName+"_"+colCol+":"+coloName)
                    ax1.grid(axis='x')
                    plt.xticks(np.arange(zdf.shape[1]),labels=zdf.columns, rotation='vertical', fontsize=7)
                    if SAVEfIGS:
                        fig = ax1.get_figure()
                        fig.savefig(save(0,BX+"_barplots",binCol+":"+binName+"_"+colCol+":"+coloName,typ="png"),bbox_inches='tight')
                    else:
                        plt.show()

                #barx += mean.shape[0]
    if piechart:
        for k in range(a.shape[0]):
            f,ax = plt.subplots()
            ax.pie(a[k,:],autopct='%.0f%%', labels=colors)
            plt.title(bins[k])
            ax.axis('equal')
            plt.show()

    colorDict = {}
    pc = "n"
    if pc == "y":
        for i in range(a.shape[1]):
            #print(colors[i])
            colorDict[colors[i]] = input("color:")
    for k,array in enumerate([a,b,]): #enumerate([a,b,c,d]):
        nam = ["raw","xnorm","clrnorm","bothnorm"][k]
        a = array.copy()
        fig, ax = plt.subplots(figsize=(15+len(bins)/5,10))
        shape = a.shape
        #print("a",a)
        height = np.zeros(shape[0])

        for i in range(shape[1]):

            try:
                AX=ax.bar(bins,a[:,i],label=colors[i],bottom=height,color=colorDict[colors[i]])
            except Exception as e:
                AX=ax.bar(bins,a[:,i],label=colors[i],bottom=height)
            for j,rect in enumerate(AX):
                nh = a[j,i]#rect.get_height()
                plt.text(rect.get_x()+rect.get_width()/2,height[j]+nh/2,round(nh,2))

            height += a[:,i]
        plt.xticks(rotation = 85)
        plt.xlabel(binCol)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
        if SAVEfIGS:
            fig = ax.get_figure()
            fig.savefig(save(0,BX+"_barplots",binCol+"_"+colCol+"_"+nam,typ="png"),bbox_inches='tight')
        plt.show()



    daf = pd.DataFrame(data=b,index=bins,columns=colors).transpose()
    f, ax = plt.subplots(figsize=(10+daf.shape[1]/5, 10+daf.shape[0]/5))
    sns.heatmap(daf,xticklabels=daf.columns,yticklabels=daf.index,center=np.mean(daf.values),annot=True)
    plt.show()
    return(df,oobs,dfxy)


'''
#processing
'''

def clusterMeans(DF,obs,dfxy):
    clusterA = []
    try:
        ucl = np.unique(obs.loc[:,'Cluster'].values)
    except:
        DF,obs,dfxy=obCluster(DF, obs, dfxy)
        ucl = np.unique(obs.loc[:,'Cluster'].values)
    nClusters = len(ucl)
    clusterA = np.zeros((nClusters,DF.shape[1]))
    #print("ucl",ucl)
    for i,c in enumerate(ucl):
        cl = DF.loc[obs['Cluster'] == c,:]
        try:
            markerMeans = np.mean(cl.values,axis = 0)
        except:
            markerMeans = np.mean(cl.values.astype(float),axis = 0)
        clusterA[i,:] = markerMeans
    #print(clusterA,"clusterA")
    return(clusterA,ucl,obs)

def obCluster(df,obs,dfxy):
    obs = obs.astype(str)
    #for i,ob in enumerate(obs.columns):
        #print(i,ob)
    #ch = int(input("cluster by?"))
    ch = -1
    chob = obs.columns[ch]
    clusters = obs[chob].copy()
    uobs = np.unique(obs[chob].values)
    for i,uo in enumerate(uobs):
        #print(i,":",uo)
        clusters.loc[obs[chob]==uo] = uo
    obs["Cluster"] = list(clusters)
    return(df,obs,dfxy)

'''
def s ave1(df,obs,dfxy,tag):
    #tag = input("filename?")
    df.to_csv(tag+"_df.csv")
    dfxy.to_csv(tag+"_dfxy.csv")
    obs.to_csv(tag+"_obs.csv")
    return(df,obs,dfxy)

def onlyPrimaries(df,obs,dfxy):
    primaries = []
    df = doPart(df)
    combd = []
    for biom in df.columns:
        #print(biom)
        if "nuclei_" in biom or "cell_" in biom:
            primaries.append(biom)
        elif "comb" in biom:
            #print(biom,"comb")
            combd.append(biom.split("_")[0])
    #print(combd)
    for biom in df.columns:
        if type(fillMap(biom)) == type(None):
            continue
        cond = 0
        for stem in combd:
            if stem in biom and "comb" not in biom:
                cond = 1
        if cond == 0:
            primaries.append(biom)
    print(primaries)
    df = df.loc[:,primaries]
    return(df,obs,dfxy)
'''

def autoleiden(df,obs,dfxy,target,res=1.306484375,incr = .33,name=None):
    print("autoLeiden",target)
    ncl = 99
    tes = []
    ret = 0
    if name:
        Name = "Leiden_" + str(target)+"_"+name
    else:
        Name = "Leiden_" + str(target)
    while ncl != target:
        if ret > 100:
            target -= 1
            ret = 0
        #print("!! running with res",res)
        adata = anndata.AnnData(df,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        obs.loc[:,Name] = ""
        obs.loc[:,Name] = adata.obs["Cluster"].astype(str)
        ncl = len(list(obs.loc[:,Name].unique()))
        #print("got",ncl,"clusters! target: ",target)
        tes.append(res)
        ret += 1
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
    return(df,obs,dfxy)



def leiden(df,obs,dfxy, ress = [.1,.3,.5,.7,.9]):
    #df,obs,dfxy,cdf = DFs[0],DFs[1],DFs[2],DFs[3]
    #print(all(obs.index==df.index),"all index the same\nrunning Leiden")
    #res = float(input("recluster with resolution:"))

    for res in ress:
        adata = anndata.AnnData(df,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        obs.loc[:,"Leiden_" + str(res)] = ""
        obs.loc[:,"Leiden_" + str(res)] = adata.obs["Cluster"].astype(str)
        #print(obs.loc[:,"Leiden_" + str(res)].unique(),len(list(obs.loc[:,"Leiden_" + str(res)].unique())),"n clusters in res!!!!",res)
    return(df,obs,dfxy)

def kmeans(df,obs,dfxy,ncls = [4,5,6,7,8,9,10]):

    for ncl in ncls:
        km = KMeans(n_clusters=ncl)
        km.fit(df)
        obs["Kmeans_"+str(ncl)] = km.labels_
    return(df,obs,dfxy)

def gmm(df,obs,dfxy,ncls=[10,20]):
    #ctypes = ['full','tied','diag','spherical']
    for nClusters in ncls:
        gmm = GMM(n_components=nClusters).fit(df)
        obs["GMM_"+str(nClusters)] = gmm.predict(df)
    return(df,obs,dfxy)



def receptorType(df,obs,dfxy):

    obs["receptors expressed"] = ''
    key1 = obs.loc[:,'Primary Celltype SVM_phenotype'] == '3 tumor'
    sdf,sobs,sxy = df.loc[key1,:],obs.loc[key1,:],dfxy.loc[key1,:]
    #print(sobs.loc[:,'SVM_phenotype'].unique())
    toCheck = sorted(["AR_","ER_","PgR_","HER2_"])
    for phen in sobs.loc[:,'SVM_phenotype'].unique():
        key = sobs.loc[:,'SVM_phenotype'] == phen
        for tc in toCheck:
            if tc in phen:
                sobs.loc[key,'receptors expressed'] += tc


    sobs.loc[sobs.loc[:,'receptors expressed'] == '','receptors expressed'] = 'TN (none)'
    print('RECEPTORTYPE',sobs.loc[:,'receptors expressed'].unique())
    obs.loc[key1,'receptors expressed'] = sobs.loc[:,'receptors expressed']
    return(df,obs,dfxy)

def neighborhoodType(df,obs,dfxy):
    zdf,zobs,zxy = zscorev(df,obs,dfxy)
    for col in df.columns:
        cols = []
        if "neigh" in col:
            cols.append(col)
            title = cols[0].split("_")[0]+"_high"
            zdf,obs,zxy = phenotype(zdf,obs,zxy,cols=cols,title=title,default="no")
            for uch in obs.loc[:,title].unique():
                if uch != "no":
                    key = obs.loc[:,title] == uch
                    obs.loc[key,title] = "yes"

    obs = obs.astype(str)
    for col in df.columns:
        if "endo" in col:
            c1 = col
        if "immune" in col:
            c2 = col

    df["endothelial_in_50um"] = df.loc[:,c1]
    df["immune_in_50um"] = df.loc[:,c2]
    df = df.drop(c1,axis=1)
    df = df.drop(c2,axis=1)
    return(df,obs,dfxy)

def phenotype(df,obs,dfxy,res = 1,cols = None,title="phenotype",default=""):
    print('\n',title)
    if type(cols) == type(None):
        cols = list(df.columns)

    '''
           ['Ecad_', 'CK7_', 'CK8_', 'CK19_', 'CK5_', 'CK14_', 'CK17_', 'MUC1_',
    'CD44_', 'AR_', 'ER_', 'PgR_','HER2_', 'EGFR_', 'CoxIV_', 'pS6RP_', 'H3K4_',
    'H3K27_', 'pHH3_', 'Ki67_', 'PCNA_', 'pRB_', 'CCND1_', 'CD45_', 'CD4_',
    'CD68_', 'CSF1R_', 'PD1_', 'FoxP3_', 'GRNZB_', 'PDPN_', 'CD31_',
    'CAV1_', 'ColIV_', 'ColI_', 'aSMA_', 'Vim_', 'gH2AX_', 'BCL2_',
    'nuclei_area', 'nuclei_eccentricity', 'LamAC_', 'RAD51_'],title="phenotype",default=""):
    '''
    print('phenotype, zscored data reqd')
    #print(df.columns,cols)
    #df = df.loc[:,cols]
    nc = []
    for col in cols:
        if col in df.columns:
            nc.append(col)
    cols = nc
    obs[title]  = default
    for col in cols:

        key = df.loc[:,col] > res
        print(col,'notyping',key.sum())
        obs.loc[key,title] += col
    return(df,obs,dfxy)





def autotype(df,obs,dfxy,chanT=True,name="autoCellType res: ",res=None): #the old version that keeps more information is in cmifAnalysis36
    roundThresh = [1500,1250,1000,750]
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP','CD90'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    odf = df.copy()
    if not res:
        res= float(input("number of standard deviations above mean required to count as +"))
    #chanT = False
    if chanT != False:
        if chanT == True:
            chanT = True
            key2 = pd.DataFrame(data=np.ones_like(odf),columns=odf.columns,index=odf.index)
            means = df.mean(axis=0)
            sds = df.std(axis=0)
            zSer = pd.Series(index = df.columns,data=means+sds*res)
            #print(zSer)
            for i,roun in enumerate(biomRounds):
                rawThresh = roundThresh[i]
                for bIm in roun:
                    for bim in df.columns:
                        if bIm+"_" in bim:
                            key2.loc[:,bim] = 0
                            key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
        elif input("check if threshold above channel threshold (non-z-score) (y)") == 'y':
            chanT = True
            key2 = pd.DataFrame(data=np.zeros_like(odf),columns=odf.columns,index=odf.index)
            means = df.mean(axis=0)
            sds = df.std(axis=0)
            zSer = pd.Series(index = df.columns,data=means+sds*res)
            #print(zSer)
            for i,roun in enumerate(biomRounds):
                rawThresh = roundThresh[i]
                for bIm in roun:
                    for bim in df.columns:
                        if bIm in bim:
                            key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
                            #bimT = zSer.loc[bim]
                            #key2.loc[odf.loc[:,bim]>bimT,bim] = 1
                            #print("ding")
        else:
            chanT = False

    #print(list(key2.iloc[:,0]),"ar key")
    obs[name+str(res)] = " "
    #if input("zscore?") == "y":
    #print(df)
    df,obs,dfxy = zscorev(df,obs,dfxy)
    #print(df)
    mapp = {}
    toThresh = []
    for biom in df.columns:
        if "neigh" in biom:
            continue
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    '''
    toThresh = list(mapp.keys())
    #others = ["Ki67", "PCNA", "pHH3","pRB","ER","PgR","AR","HER2","Fox","GRNZB","aSMA","Vim","VIM","ColI","PD1"] #CAV
    others = df.columns
    for biom in df.columns:
        for o in others:
            if o in biom:
                toThresh.append(biom)
    '''
    toThresh = list(df.columns)
    for biom in toThresh:
        if chanT:
            key3 = key2[biom] == 1
            key = df[biom]>res
            #print(biom,any(key),any(key3),any(key & key3))
            obs.loc[key & key3,name+str(res)] += biom + " "
        else:
            key = df[biom]>res
            obs.loc[key,name+str(res)] += biom + " "
    obs = parseTypes(df,obs,dfxy,column=name+str(res))
    #print(obs[name+str(res)].unique(),"uobs")
    obs = parseSecondary(df,obs,dfxy,column=name+str(res))
    #if input("keep z-scoring?") == "y":
        #return(df,obs,dfxy)
    return(odf,obs,dfxy)


def autotype1(df,obs,dfxy,chanT=True,res= float(1)): #the old version that keeps more information is in cmifAnalysis36
    roundThresh = [1500,1250,1000,750]
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    odf = df.copy()
    #float(input("number of standard deviations (or raw intensity if not zscored) above mean required to count as +"))
    #chanT = False
    if chanT != True:
        if input("check if threshold above channel threshold (non-z-score) (y)") == 'y':
            chanT = True
    key2 = pd.DataFrame(data=np.zeros_like(odf),columns=odf.columns,index=odf.index)
    if chanT:
        means = df.mean(axis=0)
        sds = df.std(axis=0)
        zSer = pd.Series(index = df.columns,data=means+sds*res)
        #print(zSer)
        for i,roun in enumerate(biomRounds):
            rawThresh = roundThresh[i]
            for bIm in roun:
                for bim in df.columns:
                    if bIm in bim:
                        key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
    else:
        key2 = 1


    #print(list(key2.iloc[:,0]),"ar key")
    obs["autoCellType res: "+str(res)] = " "
    #if input("zscore?") == "y":
    #print(df)
    df,obs,dfxy = zscorev(df,obs,dfxy)
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
    others = ["Ki67", "PCNA", "pHH3","pRB","ER","PgR","AR","HER2","Fox","GRNZB","aSMA","Vim","VIM","ColI","PD1"] #CAV
    for biom in df.columns:
        for o in others:
            if o in biom:
                toThresh.append(biom)
    for biom in toThresh:
        if chanT:
            key3 = key2[biom] == 1
            key = df[biom]>res
            #print(biom,any(key),any(key3),any(key & key3))
            obs.loc[key & key3,"autoCellType res: "+str(res)] += biom + " "
        else:
            key = df[biom]>res
            obs.loc[key,"autoCellType res: "+str(res)] += biom + " "
    obs = parseTypes(df,obs,dfxy,column="autoCellType res: "+str(res))
    #print(obs["autoCellType res: "+str(res)].unique(),"uobs")
    obs = parseSecondary(df,obs,dfxy,column="autoCellType res: "+str(res))
    #if input("keep z-scoring?") == "y":
        #return(df,obs,dfxy)
    return(odf,obs,dfxy)


def neighborhoodFractions(df,obs,dfxy,radii=[50/.325],tcol='SVM_primary',keyL = ["1 endo","2 imm"],tot = True):
    ch = list(obs.columns).index(tcol)
    uch = obs.loc[:,tcol].unique()
    goodsts = []
    for uc in uch:
        for ks in keyL:
            if ks in uc and uc not in goodsts:
                goodsts.append(uc)
    print(goodsts,"goodsts")
    for radius in radii:
        for uc in goodsts:
            df[uc+"_"+tcol+"_neighbors_"+str(radius*.325)] = 0
            #print(df.columns)
        for us in obs["slide_scene"].unique():
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            for i in range(tdfxy.shape[0]):
                ind = tdfxy.index[i]
                if i % 1000 == 1:
                    print(i/tdfxy.shape[0]*100,"% done with",us)
                #neighbors = []
                x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
                distanceV = ((x-nx)**2+(y-ny)**2)**.5
                key = distanceV < radius
                neighbors = tobs.loc[key,:]
                neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
                nnei = neighbors.shape[0]
                if nnei > 1:
                    #ngoodsts = neighbors.loc[:,obcol]
                    for uc in goodsts:
                        inNei = neighbors.loc[neighbors.loc[:,tcol] == uc,tcol]
                        if tot:
                            df.loc[ind,uc+"_"+tcol+"_neighbors_"+str(radius*.325)] = inNei.shape[0]
                        else:
                            df.loc[ind,uc+"_"+tcol+"_neighbors_"+str(radius*.325)] = inNei.shape[0]/nnei
                else:
                    for uc in goodsts:
                        df.loc[ind,uc+"_"+tcol+"_neighbors_"+str(radius*.325)] = 0
    #print(df.columns)
    return(df,obs,dfxy)



def parseTypes(df,obs,dfxy,column="none"):
    #if column == 'none':
        #ch,uch = obMenu("column to apply types to")
        #column = obs.columns[ch]
    mapp = {}
    for biom in df.columns:
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    #print(mapp)
    mapp = {k: v for k, v in sorted(mapp.items(), key=lambda item: item[1])}
    #print(mapp)
    obs["Primary Celltype "+column] = "5 stromal"
    for biom in mapp.keys():
        keyCol = obs[column].str.contains(biom)
        #print(list(keyCol))
        unasKey = obs["Primary Celltype "+column] == "5 stromal"
        obs.loc[keyCol & unasKey,"Primary Celltype "+column] = mapp[biom]
    return(obs)

def chanThresh(df,obs,dfxy):
    roundThresh = [1500,1250,1000,750]
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]


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
    '''
    key2 = pd.DataFrame(data=np.ones_like(df),columns=df.columns,index=df.index)

    for i,roun in enumerate(biomRounds):
        rawThresh = roundThresh[i]
        for bIm in roun:
            for bim in df.columns:
                if bIm+'_' in bim:
                    key2.loc[df.loc[:,bim]<=rawThresh,bim] = 0
    '''
    #print(key2)
    return(key2)


def aiSecondary(df,obs,dfxy,typCol="SVM_primary",title="SVM_phenotype"):
    oodf = df.copy()
    #ncn = typCol.split("_")[0]+"_secondary"
    obs[title] = ""
    ctk = chanThresh(df,obs,dfxy)
    #print(df.shape,ctk.shape)
    df,obs,dfxy = zscorev(df,obs,dfxy)
    for col in ctk.columns:
        colk = ctk.loc[:,col] != 1.0
        #print(colk.sum(),"colk sum")
        df.loc[colk,col] = 0
    sbioms = [['Ecad_', 'CK7_', 'CK8_', 'CK19_', 'CK5_', 'CK14_', 'CK17_', 'MUC1_',
           'CD44_', 'AR_', 'ER_', 'PgR_','HER2_', 'EGFR_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"],
              ['CD45_', 'CD4_', 'CD68_',  'PD1_', 'FoxP3_', 'GRNZB_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"],
              ['CD31_','CAV1_', 'ColIV_', 'ColI_', 'aSMA_', 'Vim_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"]]
    k1 = obs.loc[:,typCol] == '3 tumor'
    k2 = obs.loc[:,typCol] == '2 immune'
    k3 = k1 & k2
    for i,key in enumerate([k1,k2, ~k3]):
        scols = sbioms[i]
        sdf,sobs,sxy = df.loc[key,:],obs.loc[key,:],dfxy.loc[key,:]
        sdf,sobs,sxy = phenotype(sdf,sobs,sxy,cols=scols,title=title)
        obs.loc[key,title]+=sobs[title]
    obs["Primary Celltype "+title] = obs.loc[:,typCol]
    obs = parseSecondary(df,obs,dfxy,title)

    return(oodf,obs,dfxy)

def aiSecondary1(df,obs,dfxy,typCol="SVM_primary",title="SVML_phenotype"):
    oodf = df.copy()
    obs[title] = ""
    ctk = chanThresh(df,obs,dfxy)
    #print(df.shape,ctk.shape)
    df,obs,dfxy = zscorev(df,obs,dfxy)
    for col in ctk.columns:
        colk = ctk.loc[:,col] != 1.0
        #print(colk.sum(),"colk sum")
        df.loc[colk,col] = 0
    sbioms = [['Ecad_', 'CK7_', 'CK8_', 'CK19_', 'CK5_', 'CK14_', 'CK17_', 'MUC1_',
           'CD44_', 'AR_', 'ER_', 'PgR_','HER2_', 'EGFR_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"],
              ['CD45_', 'CD4_', 'CD68_',  'PD1_', 'FoxP3_', 'GRNZB_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"],
              ['CD31_','CAV1_', 'ColIV_', 'ColI_', 'aSMA_', 'Vim_','nuclei_area', 'nuclei_eccentricity',"Ki67_","pHH3_","PCNA_"]]

    nsb = []
    for cols in sbioms:
        nc = []
        for col in cols:
            if col in df.columns:
                nc.append(col)
        nsb.append(nc)
    sbioms = nsb

    k1 = obs.loc[:,typCol] == '3 tumor'
    k2 = obs.loc[:,typCol] == '2 immune'
    k3 = ~k1 & ~k2
    for i,key in enumerate([k1,k2,k3]):
        #print(k3.sum(),~k3.sum())
        name = ["tumor","immune","other"]
        nam = name[i]
        scols = sbioms[i]
        sdf,sobs,sxy = df.loc[key,scols],obs.loc[key,:],dfxy.loc[key,:]

        ncl = 22
        sdf,sobs,sxy = autoleiden(sdf,sobs,sxy,ncl,name=nam)
        col1 = "Leiden_"+str(ncl)+"_"+nam
        ch = list(sobs.columns).index(col1)
        tdf,tobs,txy = clag(sdf,sobs,sxy,ch,sobs.loc[:,col1].unique(),z=True)
        tdf,tobs,txy = phenotype(tdf,tobs,txy,cols=scols,title=title)
        for col in tobs.columns:
            if title in col:
                #print(col,aobs.loc[:,col].unique(),"!!")
                sobs[col] = ""
                for uc in tobs.index:
                    key0 = sobs.iloc[:,ch] == uc
                    sobs.loc[key0,col] = tobs.loc[uc,col]
            if col not in obs.columns:
                obs[col] = ""
            obs.loc[key,col] = sobs.loc[:,col]

    obs["Primary Celltype "+title] = obs.loc[:,typCol]
    obs = parseSecondary(df,obs,dfxy,title)
    return(oodf,obs,dfxy)


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


    proL = ["Ki67","pHH3","PCNA"]#"pRB"
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3+ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68"] #ADD CSF1R?
    Hl = ['ER_', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_"]
    for typ in uch:
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






def fillMap(biom):
    bTypes = [["1 endothelial",["CD31","CAV"]],
              ["2 immune",["CD4_","CD45","CD3_","CD68"]],
              ["3 tumor",["CK","Ecad","MUC1",'EGFR',"HER"]],
              ["4 active fibroblast",["aSMA","Vim","VIM","ColI_","CD90"]]]
    for typeA in bTypes:
        for stem in typeA[-1]:
            if "CD44" in biom or "in radius" in biom or "neighbors" in biom:
                return(None)
            if stem in biom:
                return(typeA[0])





def zscorev(df,obs,dfxy):
    df,obs,dfxy = zscore(df,obs,dfxy,ax=0)
    return(df,obs,dfxy)

def zscoreh(df,obs,dfxy):
    df,obs,dfxy = zscore(df,obs,dfxy,ax=1)
    return(df,obs,dfxy)

def zscore(df,obs,dfxy,ax=None):
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
        #PROCESS.append("z-score_(v)")
    if ax == 1:
        for i in range(shape[0]):
            col = vals[i,:].tolist()
            zCol = zScoreL(col)
            newA[i,:] = zCol
        #PROCESS.append("z-score_(h)")
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

stt = time.time()

if __name__ == '__main__':
    main()
    #print((time.time()-stt)/60," minute runtime main1")
    #main2()
    print((time.time()-stt)/60," minute runtime total")