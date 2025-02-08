# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:57:00 2024

@author: youm
"""

import time
GST = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore as ZSC #only for calculating correlation


SAVENAME = 'test_250130'

COL = 'subset_slide_scene' #uses each unique value
SHOW = True
SAVE = False
NFIGS = 10
testDFs = 1
MAXLOSS = -9#10**-4
STIME = 0
num_columns = 30 #no '_'
feature_dim = 1000 #cells per slide_scene

LR_ = [.0001]
#num_columns_ = [10,30]#000 #11
#feature_dim_ = [50,150] #20000
num_heads_ = [1]  #feature_dim must be divisible by num_heads
max_epochs_ = [10000]



Results = []


#index is run number
#obs have each variable 'LR' cat has ['.0001,'.'0002']
'''
total = 1
for i,lis in [LR_,num_columns_,feature_dim_,num_heads_]:
    total *= len(list)
for i in range(total):
    toTest_.append([])
'''


import copy

toTest = LR_
for i in range(len(toTest)):
    toTest[i] = [toTest[i]]

for i,lis in enumerate([num_heads_,max_epochs_]):
    print(i,lis)
    nl = toTest.copy()
    for ii in range(len(lis)-1):
        toTest = toTest + copy.deepcopy(nl)
    print(toTest)
    for j,el in enumerate(lis):
        start = j*len(nl)
        end = (j+1)*len(nl)
        for ind in range(start,end):
            print(ind,toTest[ind],[el])
            toTest[ind] += [el]
            #print(toTest,'\n')

print(toTest,len(toTest),'combinations of LR_, num_columns_, feature_dim_, num_heads_,max_epochs to test \n----------------------------------------\n')


#try multi-layer neural network
#Consider  confusion matrix and specificity
#Don't have enough exposure to the positive case - not representative, not enough positive cells
#deal with 'class imbalance' problem
#figure out overfitting
#Check model architecture and hyperparameters

class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ff_hidden_dim, feature_dim)
        )
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [num_columns, batch_size, feature_dim] (for nn.MultiheadAttention)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x


class TransformerPredictor(nn.Module):
    def __init__(self, num_layers, feature_dim, num_columns, num_heads, ff_hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_columns, feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, ff_hidden_dim, dropout_prob)
            for _ in range(num_layers)
        ])
        self.fc_output = nn.Linear(feature_dim, feature_dim)  # Final prediction layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, num_columns, feature_dim]

        batch_size, num_columns, feature_dim = x.size()

        # Prepare input for multihead attention (requires shape [num_columns, batch_size, feature_dim])
        x = x.permute(1, 0, 2)

        # Pass through transformer layers
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Pool across the sequence dimension (num_columns)
        x = x.mean(dim=0)  # [batch_size, feature_dim]

        # Final prediction layer
        x = self.fc_output(x)
        predictions = self.sigmoid(x)

        return predictions  # Shape: [batch_size, feature_dim]

# Training Routine
def train_model(data, targets, num_heads=2, num_layers=3, ff_hidden_dim = 200,
                dropout_prob = .1, max_epochs=100, lr=1e-4, log_interval=10):
    print(data.size())
    print(targets.size())
    input('training sizes!')
    feature_dim = data.size()[2]
    num_columns = data.size()[1]
    model = TransformerPredictor(num_layers, feature_dim, num_columns, num_heads, ff_hidden_dim, dropout_prob)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Mean Squared Error loss
    expression = data[:, 0, :].reshape(-1).detach()
    print
    for epoch in range(1, max_epochs + 1):
        model.train()

        # Forward pass
        predicted_thresholds = model(data)
        #print(predicted_thresholds.size(),'pthresh s')

        # Compute loss
        loss = loss_fn(predicted_thresholds, targets)
        if loss.item() < MAXLOSS:
            print("Training complete in",epoch,"epochs!")
            return(predicted_thresholds)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            #print(loss.item() < MAXLOSS,loss.item(),MAXLOSS)
            print(round((time.time()-STIME)/60,3),'minutes elapsed')
            show(predicted_thresholds.detach(),expression,targets.detach())
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f}")
            print(round((time.time()-STIME)/60,3),'minutes elapsed\n\n')

    print("Training complete!")
    return(predicted_thresholds,model)



def scatterplot(sheet,i1 = 0, i2 = 1,xn=None,yn=None,vl=None):
    plt.scatter(sheet[:,i1],sheet[:,i2],alpha=.1)
    plt.xlabel(xn)
    plt.ylabel(yn)
    plt.vlines(vl,sheet[:,i2].min(),sheet[:,i2].max())

    plt.show()

def show(output,expression,targets,title=None):
    #print(len(targets),len(output),'x,y')
    #print(targets.size(),'targets size show')
    #print(output.size(),'output size show')
    #print(expression.size(),'expression size show')
    #input()
    output = output.view(-1,1)
    targets = targets.view(-1,1)
    colors = ['red' if t == 1 else 'blue' for t in targets.numpy()]

    plt.figure(figsize=(8, 6))
    plt.scatter(expression.numpy(), output.numpy(), c=colors, alpha=0.6, edgecolors='k')
    plt.xlabel("Expression")
    plt.ylabel("Output")
    plt.title(title)
    plt.show()


def show1(targets,output,title=None):
    #print(len(targets),len(output),'x,y')

    plt.scatter(targets,output)
    plt.title(title)
    plt.show()


def load(fold,num_columns):

    DFs = [] #list of dataframes (one per slide)
    #targets = {} #dict of biom_slide : thresh
    targets = []
    for file in os.listdir(fold):
        #if 'MIT' not in file:
        #    continue
        if '_df.csv' in file:
            stem = file.split('_df.csv')[0]
            print(stem)
            for f2 in os.listdir(fold):
                if stem in f2 and '_obs.csv' in f2:
                    df0 = pd.read_csv(fold+'/'+file,index_col=0)
                    df0 = clean(df0) #removes nuc area and eccentricity
                    obs0 = pd.read_csv(fold+'/'+f2,index_col=0)
                    for slide in obs0.loc[:,COL].unique():
                        key = obs0.loc[:,COL] == slide
                        df = df0.loc[key,:]
                        #df = df.sort_values(by = COL) #
                        df = normalize(df)
                        DFs.append(df)
                        obs = obs0.loc[key,:]
                        ta = getTargets(df,obs,slide)
                        for ii,tl in enumerate(ta):
                            if ii >= num_columns: #trim cols past num_columns for answers
                                break
                            targets.append(tl)
                            print(slide,len(targets))
                        #targets.update(td)
                        #targets.append(td)
        return(DFs,targets)

def clean(df):
    badSts = ['_area','_eccentr','DAPI1_','LamAC','CD4_','CD8_']
    os = df.shape[1]
    for col in df.columns:
        for bs in badSts:
            if bs in col:
                print('removing..',col)
                df = df.drop(col,axis=1)
    print(os-df.shape[1],'cols removed!')
    return(df)

def normalize(df,mx = 1000000):
    df = df.where(df < mx,mx)
    df = np.log(df+1)
    return(df)


def getTargets(df,obs,slide):
    td = [] #{}
    for biom in df.columns:
        out = pd.Series(np.zeros(obs.shape[0]),index=obs.index)
        bn = biom.split('_')[0]+'_'
        key = obs['Manual Celltype'].astype(str).str.contains(bn)
        out.loc[key] = 1
        td.append(out)
        #print(out.sum(),'out sum')
    return(td)




def makeData(DFs,num_columns,feature_dim):
    data = []
    blank = np.zeros((feature_dim,num_columns))
    #print(blank.shape)

    for temp,df in enumerate(DFs):
        if df.shape[0] > feature_dim:  #trim data that's bigger than model
            df = df.iloc[:feature_dim]
        zdf = df.apply(ZSC)
        for ii,biom in enumerate(df.columns): #biom is first column in sheet
            if ii >= num_columns:             #trim cols that are past num_cols
                break

            sheet = blank.copy()
            sheet[0:df.shape[0],0] = df.loc[:,biom] #data fills up subset of sheet
            cors = []
            for other in df.columns:
                cor = np.dot(zdf.loc[:,biom],zdf.loc[:,other]) #integer total correlation
                cors.append(cor)
            bioms = list(df.columns)
            i = 1   #already has one column (biom to threshold)
            while len(cors) > 0:
                if i >= num_columns:
                    break              #trim columns that are bigger than model, keeping the most relevant
                mind = cors.index(min(cors))
                bm = bioms[mind]      #bm is other column
                #print(biom,min(cors),bm)
                sheet[0:df.shape[0],i] = df.loc[:,bm].copy()

                if i == 1:
                    print('scatter!!')
                    scatterplot(sheet,0,1,biom,bm)
                i += 1
                #print(cors,mind)
                cors.pop(mind)
                bioms.pop(mind)
            data.append(sheet)
            #print(sheet)
            #print(len(data),biom,temp)
            #input()
    return(torch.tensor(np.array(data),dtype=torch.float32).permute(0, 2, 1))

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


def score(targets,predicted_thresholds):
    score = (targets.detach().numpy()-.5) * (predicted_thresholds.detach().numpy()-.5)
    #print(score,'score')
    #osc = np.zeros_like(score)
    osc = np.where(score > 0,1,0)
    #print(osc,'osc')
    return(round(np.mean(osc)*100,1))



# Example usage
if __name__ == "__main__":

    # Example parameters
    #print('try to predict fraction of cells above threshold')

    #for i in range(len(toTest)): move here to test feature_dim_ and num_columns_
    #STIME = time.time()
    #print(STIME,'!!!!')
    #tt = toTest[i]
    #LR,num_columns,feature_dim,num_heads,max_epochs = tt[0],tt[1],tt[2],tt[3],tt[4]

    DFs,targets = load(r'C:\Users\youm\Desktop\src\transformernet_training',num_columns) #one DF per slide_scene
    #DFs,targets = load(r'D:\Classified_CSV\Classified_CSV',num_columns,style='sam')
    print(len(DFs),'DFs loaded!')
    t = np.zeros((len(targets),feature_dim))
    for i,tl in enumerate(targets):
        if len(tl) > feature_dim:
           tl = tl[:feature_dim]
           t[i,:len(tl)] = tl
    targets1 = t[-testDFs*num_columns:]
    targets = t[:-testDFs*num_columns]
    print(len(targets),'len targets')
    print(len(targets1),'len targets1')
    targets = torch.tensor(targets,dtype=torch.float32)
    targets1 = torch.tensor(targets1,dtype=torch.float32)
    #print(t)
    print(targets.size(),'targets')
    batch_size = len(DFs)
    if batch_size < testDFs:
        testDFs = batch_size


    data = makeData(DFs[:-testDFs],num_columns,feature_dim)
    tr_expression = data[:, 0, :].reshape(-1).detach()
    test_data = makeData(DFs[-testDFs:],num_columns,feature_dim)
    te_expression = test_data[:, 0, :].reshape(-1).detach()
    print(targets)
    print(tr_expression,'training')
    #input()
    print(te_expression,'test')
    #input()
    print(data.size())
    ds = data.size()[0] * data.size()[1] * data.size()[2]
    df = []
    obs = []
    dfxy = []
    for i in range(len(toTest)):
        STIME = time.time()

        tt = toTest[i]
        runname = ','.join([str(item) for item in tt])
        print(runname,'variables')

        LR,num_heads,max_epochs = tt[0],tt[1],tt[2]
        predicted_thresholds,model = train_model(data, targets, max_epochs=max_epochs, lr=LR, log_interval=int(max_epochs/NFIGS))
        #print(predicted_thresholds)
        diff = targets-predicted_thresholds
        adiff = torch.mean((diff**2)**.5)
        print('average difference train:',adiff)
        trac = score(targets,predicted_thresholds)
        print(trac,'training accuracy')
        trtime = round((time.time()-STIME)/60,3)
        print(trtime,'minutes elapsed')
        show1(targets.detach(),predicted_thresholds.detach(),title=runname+'\n'+str(trtime)+' '+str(trac))
        show(predicted_thresholds.detach(),tr_expression,targets.detach(),title=runname+'\n'+str(trtime)+' '+str(trac))

        predicted_thresholds = model(test_data)
        diff = targets1-predicted_thresholds
        tadiff = torch.mean((diff**2)**.5)
        print('average difference test:',tadiff)
        teac = score(targets1,predicted_thresholds)
        #pd.DataFrame([targets1.view(-1,1,1).detach().numpy(),predicted_thresholds.view(-1,1,1).detach().numpy()]).to_csv('temp.csv')

        show1(targets1.detach(),predicted_thresholds.detach(),title=runname+'\ntest: '+str(teac)+'%')
        show(predicted_thresholds.detach(),te_expression,targets1.detach(),title=runname+'\ntest: '+str(teac)+'%')
        print(teac,'test accuracy')
        df.append([trtime,trac,teac,ds])
        obs.append([runname] + runname.split(',')+[num_columns,feature_dim])
        dfxy.append([trtime,teac])
        tote = round((time.time()-GST)/60,2)
        print(tote,'minutes elapsed total')
        print(round((i+1)/len(toTest)*100,1),'% done')
        #time/done = x/remaining
        print(round((1-(i+1)/len(toTest))*tote/((i+1)/len(toTest)),1),'minutes remaining')
        print('\n---------------------------------------------------------------------------\n')

    SAVENAME = SAVENAME+'_'+str(ds)
    obs = pd.DataFrame(obs,columns = ['run name','Learning rate','n heads','epochs','n columns','n features'])
    print(obs)
    obs.index = obs.loc[:,'run name']
    obs['slide_scene'] = SAVENAME
    df = pd.DataFrame(df,columns=['training time','training accuracy','test accuracy','data size'],index = obs.index)
    dfxy = pd.DataFrame(dfxy,columns=['training_time_x','test_score_y'],index=obs.index)
    if SAVE:
        save(df,obs,dfxy,filename=SAVENAME)
