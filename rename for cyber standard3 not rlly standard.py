
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:16:21 2022

@author: youm
"""
#0
import pandas as pd
import os


FOLD = r'T:\Cyclic_Images\cmIF_2023-11-15_TMAs\U54-TMA-18'#r'T:\Cyclic_Images\cmIF_2023-11-15_TMAs'

#f0 = r'T:\Cyclic_Images\cmIF_2023-10-04_KLF4-2\R8\BC-TMA1-14\splitscenes'
#d0 = r'T:\Cyclic_Images\cmIF_2023-10-04_KLF4-2\R8/_BC-TMA1-14_Coresorting_R8.xlsx'
#f1 = r'T:\Cyclic_Images\cmIF_2023-10-04_KLF4-2\R8\BR301-121\splitscenes'
#d1 = r'T:\Cyclic_Images\cmIF_2023-10-04_KLF4-2\R8/_BR301-121_Coresorting_R8.xlsx'

#dpaths =
folds = ['splitscenes','stitched','TIFF'] #script works assuming the first col has new scene IDS, other cols have R0 R1 as col headers
#folds = [f0]


reqd = ['.']


TEST = False






renames = []
dpat = r'T:\Cyclic_Images\cmIF_2023-11-15_TMAs\IY coresorting copy.xlsx'

xl = pd.ExcelFile(dpat)
SN = xl.sheet_names

for ii in range(len(folds)):
    #print(sn)
    #continue
    fold = FOLD+'/'+folds[ii]
  #+'/'+dpaths[ii]
    for i in range(5):
        if i != 4:
            #print('only looking at last sheet')
            continue
        sn = SN[i]
        df = pd.read_excel(dpat,sheet_name=i).fillna('999999').astype(str)
        df = df.where(df != 'x','99999')
        print(df)
        for file in sorted(os.listdir(fold)):
            if sn not in file:
                continue
            print(file,sn)
            switch = 1
            for rstr in reqd:
                if rstr not in reqd:
                    switch = 0
            if switch == 0:
                continue
            roun = file.split('_')[0]
            if 'Q' in roun or int(roun.split('R')[-1]) != 8:
                #print('only round 8')
                continue
            try:
                scene = file.split('cene-')[-1].split('.')[0].split('_')[0]
                #try:
                scene = int(scene)
                key = df.loc[:,roun].astype(float).astype(int) == scene
            except Exception as e:
                renames.append([file,'problem with scene: '+str(scene)+str(e)])
                print('bad scene',scene)
                continue
            #print(roun,file,fold)

            if key.sum() != 1:
                print('error with ',file,key.sum())
                renames.append([fold+'/'+file, '!key sum: '+str(key.sum())])

            else:
                ID = df.loc[key,:].values[0][0]
                print(file,ID)
                if '.czi' in file:
                    nn = file.split('cene-')[0]+'cene-' + ID +'.'+ file.split('.')[-1]
                elif '.tif' in file: #keeps channel info
                    nn = file.split('cene-')[0]+'cene-' + ID +'_'+ file.split('_')[-2]+'_'+file.split('_')[-1]
                print(nn)
            if TEST:
                renames.append([file,nn])
            print('\n')
            if not TEST:
                try:
                    os.rename(fold+"/"+file,fold+"/"+nn)
                    renames.append([file,nn])
                except Exception as e:
                    renames.append([file,e])
                    print(e)

out = pd.DataFrame(renames,columns=['old','new'])
out.to_csv(FOLD+"/renames_u54TMA_R8.csv")
