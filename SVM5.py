# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:55:50 2022

@author: youm
"""

import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import os
import pickle

TP = .1

class Data():
    def __init__(self,df,key,X,Y,Xt,Yt):
        self.df=df
        self.key=key
        self.X=X
        self.Y=Y
        self.Xt=Xt
        self.Yt=Yt



def main(df,obs,dfxy):
    X,Y = buildData(df,obs,dfxy)
    print(X,"X")
    print(Y,"Y")
    op = ["train svm","load svm to predict data"]
    fn = [trainSVM,useSVM]
    predictions = menu(op,fn,X,Y)
    plt.scatter(Y,predictions)
    plt.xticks(rotation = 85)
    plt.show()
    obs["predictions"] = predictions
    return(df,obs,dfxy)




def menu(options,functions,X,Y):
    while True:
        print("\n")
        for i,op in enumerate(options):
            print(i,op)
        try:
            print("send non-int when done (return)")
            ch = int(input("number: "))
        except:
            return(out)#X,Y)
        out = functions[ch](X,Y)


def trainSVM(X,Y,clf=None):
    if isinstance(clf,type(None)):
        clf = svm.SVC()
    clf.fit(X,Y)
    pickle.dump(clf,open("tempSVM.sav",'wb'))
    return(clf.predict(X))

def useSVM(X,Y):
    clf = pickle.load(open("tempSVM.sav", 'rb'))
    pr = clf.predict(X)
    if not isinstance(Y,type(None)):
        sheet = np.where(pr==Y,1,0)
        print(np.append([pr],[Y],axis=1))
        print("accuracy:",np.sum(sheet)/sheet.shape[0]*100,"%")
    return(pr)


def buildData(df,obs,dfxy):
    yind = pickYind(obs)
    df = makeDtype(df,float)
    X = df.values
    try:
        Y = obs.iloc[:,yind].astype(float)
        print(Y,"\n\n")
        print("numerical data detected, binarizing around the MEAN")
        nY = pd.Series(np.zeros_like(Y))
        mean = Y.mean()
        print("mean",mean)
        nY.loc[Y>mean] = 1
        print(list(nY),"Y!")
        return(X,nY)
    except Exception as e:
        print(e)
        '''
        sy = makeDtype(obs,float).iloc[:,yind]
        print(sy)
        uty = sy.unique()
        Y = np.zeros_like(sy).astype(int)
        for i,ut in enumerate(uty):
            print(ut)
            sk = sy == ut
            Y[sk] = i
        '''
        Y = obs.iloc[:,yind]
        return(X,Y)


def main2():
    df = loadData()
    print(df)
    data = makeData(df)
    clf = train(data.X,data.Y)
    pr = test(clf,data.Xt,data.Yt)


def test(clf,X,Y=None):
    pr = clf.predict(X)
    if not isinstance(Y,type(None)):
        sheet = np.where(pr==Y,1,0)
        print(np.append([pr],[Y],axis=0))
        print("accuracy:",np.sum(sheet)/sheet.shape[0]*100,"%")
    return(pr)


def train(X,Y,clf=None):
    if isinstance(clf,type(None)):
        clf = svm.SVC()
    clf.fit(X,Y)
    return(clf)


def makeData(df):
    yind = pickYind(df)
    cn = df.columns[yind]
    out = df.pop(cn)

    df = dropCols(df)
    df = makeDtype(df,float)
    df[cn]=out

    print(df,df.shape)
    dice=np.random.rand(df.shape[0])
    key = np.where(dice<TP,1,0)
    data = Data(df,key,None,None,None,None)
    data.X = df.loc[key==0,:].values[:,:-1]
    data.Xt = df.loc[key==1,:].values[:,:-1]

    sy = df.iloc[:,-1]
    Yboth = np.zeros_like(sy).astype(int)
    uty = sy.unique()
    if len(uty) < 20:
        for i,ut in enumerate(uty):
            print(ut)
            sk = sy == ut
            Yboth[sk] = i
    else:
        print(uty,"too many different outputs to categorize, binarizing around mean")
        sy = sy.astype(float)
        mean = sy.mean()
        Yboth = np.where(sy>mean,1,0)
        print(list(Yboth),"yboth")


    data.Y = Yboth[key==0]
    data.Yt = Yboth[key==1]
    print(data.Y,data.Yt)
    print(data.X.shape,data.Y.shape)
    print("finished building data")
    return(data)

def makeDtype(df,dtype=float):
    print("\n",dtype)
    for i in range(df.shape[1]):
        print(df.columns[i])
        try:
            df.iloc[:,i] = df.iloc[:,i].astype(dtype)
        except:
            print('could not convert',df.columns[i])
            u = df.iloc[:,i].unique()
            if len(u) < 100:
                new = np.arange(len(u)).astype(dtype)
                print(u,new)
                for j,un in enumerate(u):
                    key = df.iloc[:,i] == un
                    df.loc[key,df.columns[i]] = new[j]
                df.iloc[:,i] = df.iloc[:,i].astype(dtype)
                print("        replaced with ints")
            else:
                df.iloc[:,i] = np.ones(df.shape[0]).astype(dtype)
                print("             replaced with ones")
        print(df.shape)
    df = df.fillna(0)
    return(df)


def dropCols(df):
    lis = [df]
    for k,d in enumerate(lis):
        print(list(d.columns))
        toRem = flexMenu(title="remove all columns containing these strings")
        print("finished flexmenu")
        if len(toRem) == 0:
            return(df)
        print("\nbefore:\n",lis[k].columns)
        dr = []
        for col in d.columns:
            for t in toRem:
                if t in col:
                    dr.append(col)
        lis[k] = tryDrop(d,dr)
        print("after:\n",lis[k].columns)
    for d in lis:
        print("\n",d.columns)
    return(lis[0])


def tryDrop(df,dropList):
    for colName in dropList:
        try:
            df = df.drop([colName],axis = 1)
        except:
            #pass
            print(colName,'not in dataframe')
    return(df)

def pickYind(df):
    while True:
        for i,col in enumerate(df.columns):
            print(i,col)
        try:
            ch=int(input("Answers column:"))
            return(ch)
        except:
            pass



def loadData():
    #df = pd.read_csv("iris.csv")
    file = navigate(r"C:\Users\youm\Desktop\old src")
    df = pd.read_csv(file,dtype=object,index_col=0)
    if df.shape[1] < 2:
       df = pd.read_csv(file,dtype=object,sep=" ")
    return(df)


def navigate(path,text="send non-int to go back one step, send 999 to return entire folder"):
    folder = sorted(os.listdir(path))
    for i,thing in enumerate(folder):
        print(i,thing)
    try:
        print("\n"+text)
        ch = int(input("access which number?"))
        if ch == 999:
            return([path])
        path = path+"/"+folder[ch]
        print(path)
    except:
        plis = path.split("/")
        plis = plis[:-1]
        path = "/".join(plis)
    return(path)

def flexMenu(title="String to include in list"):
    lis = []
    while True:
        ch=input(title+" (send blank when done): ")
        if ch == "":
            return(lis)
        lis.append(ch)
if __name__ == "__main__":
    main2()