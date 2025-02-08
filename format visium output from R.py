# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:52:39 2024

@author: youm
"""

import pandas as pd
import numpy as np

df = pd.read_csv('3011_01.csv',index_col=0).transpose()
x1 = pd.read_csv('3011_xy.csv',index_col=0) #comes first in df it seems
x2 = pd.read_csv('3011_xy_A.csv',index_col=0)

print(df,'df')
print(x1,'x1')
print(x2,'x2')
#df['imagerow'] = ''
#df['imagecol'] = ''
upDf = df.iloc[:x1.shape[0],:]
#for i in range(x1.shape[0]):
#    print(upDf.index[i],x1.index[i])

ind = pd.Series(np.arange(x1.shape[0])).astype(str)+'_D'
x1.index = ind
upDf.index = ind
d1 = pd.concat([upDf,x1],axis=1)
print(d1.columns)


DDf = df.iloc[x1.shape[0]:,:]
#for i in range(DDf.shape[0]):
#    print(DDf.index[i],x2.index[i])

ind = pd.Series(np.arange(x2.shape[0])).astype(str)+'_A'

x2.index = ind
DDf.index = ind
d2 = pd.concat([DDf,x2],axis=1)
print(d2.columns)

df = pd.concat([d1,d2],axis=0)
print(df.columns)

df.to_csv('3011_00.csv')