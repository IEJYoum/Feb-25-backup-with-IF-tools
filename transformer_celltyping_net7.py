# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:57:00 2024

@author: youm
"""


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
import time

COL = 'subset_slide_scene' #uses each unique value

MAXREPS = 1000
LR = 10**-4
MAXLOSS = 10**-3
STIME = time.time()
TRAIN = True

NFIGS = 20


#old version of ai was in tcn5
class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = feature_dim // num_heads

        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        self.fc = nn.Linear(feature_dim, feature_dim)

        # Dropout layers for attention and output
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [batch_size, num_columns, feature_dim]
        batch_size, num_columns, feature_dim = x.size()

        # Compute Queries, Keys, Values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = Q.view(batch_size, num_columns, self.num_heads, self.dim_per_head).transpose(1, 2)
        K = K.view(batch_size, num_columns, self.num_heads, self.dim_per_head).transpose(1, 2)
        V = V.view(batch_size, num_columns, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)  # Apply dropout to attention weights

        # Compute weighted sum of Values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_columns, feature_dim)

        # Concatenate heads and apply final linear projection
        output = self.fc(context)
        output = self.output_dropout(output)  # Apply dropout to the final output

        # Add residual connection and apply layer normalization
        output = self.layer_norm(output + x)

        return output, attention_weights

# Threshold Prediction Module with Weight Decay Configurable
class ThresholdPredictor(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.attention = SelfAttention(feature_dim, num_heads, dropout_prob)
        self.fc = nn.Linear(feature_dim, 1)  # Predict a single scalar per batch

    def forward(self, x):
        # x: [batch_size, num_columns, feature_dim]
        attention_output, attention_weights = self.attention(x)

        # Aggregation method (mean across columns)
        a_o = attention_output.mean(dim=1)  # [batch_size, feature_dim]

        predicted_threshold = self.fc(a_o).squeeze(-1)
        return predicted_threshold, attention_weights


# Training Routine
def train_model(model, data, targets, max_epochs=100, lr=1e-4, log_interval=10):
    #model = ThresholdPredictor(feature_dim, num_heads)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Mean Squared Error loss

    for epoch in range(1, max_epochs + 1):
        model.train()

        # Forward pass
        predicted_thresholds, attention_weights = model(data)

        # Compute loss
        loss = loss_fn(predicted_thresholds, targets)
        if loss.item() < MAXLOSS:
            print("Training complete in",epoch,"epochs!")
            return(predicted_thresholds, attention_weights)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            #print(loss.item() < MAXLOSS,loss.item(),MAXLOSS)
            print(round((time.time()-STIME)/60,3),'minutes elapsed')
            show(targets.detach(),predicted_thresholds.detach())
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f}")
            print(round((time.time()-STIME)/60,3),'minutes elapsed\n\n')

    print("Training complete!")
    return(predicted_thresholds, attention_weights)


def show(targets,output):
    #print(len(targets),len(output),'x,y')
    plt.scatter(targets,output)
    plt.show()


def load(fold):

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
                    df0 = clean(df0) #removes nuc area and eccentricity
                    obs0 = pd.read_csv(fold+'/'+f2,index_col=0)
                    for slide in obs0.loc[:,COL].unique():
                        key = obs0.loc[:,COL] == slide
                        df = df0.loc[key,:]
                        #df = df.sort_values(by = COL) #
                        df = normalize(df)
                        DFs.append(df)
                        obs = obs0.loc[key,:]
                        thresh,zinf = getThresh(df,obs,slide)
                        threshs.update(thresh)
                        zinfo.update(zinf)
        #print(threshs,'threshs')
        #print(zinfo,'zinfo')
        return(DFs,threshs,zinfo)

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
            #print(key,key.sum())
            pser = df.loc[key,biom]
            thresh = pser.min()
        #print(thresh)
        thd[biom+'_'+slide] = thresh
    return(thd,zinf)



def scatterplot(sheet,i1 = 0, i2 = 1,xn=None,yn=None,vl=None):
    plt.scatter(sheet[:,i1],sheet[:,i2])
    plt.xlabel(xn)
    plt.ylabel(yn)
    plt.vlines(vl,sheet[:,i2].min(),sheet[:,i2].max())

    plt.show()

def makeData(DFs,num_columns,feature_dim,ncols_to_trim = 3, percentile = 99, threshs = None):
    if ncols_to_trim >= num_columns:
        ncols_to_trim = num_columns - 1
    data = []
    blank = np.zeros((feature_dim,num_columns))
    #print(blank.shape)

    for temp,df in enumerate(DFs):
        if df.shape[0] > feature_dim:  #trim data that's bigger than model
            df = df.iloc[:feature_dim]
        zdf = df.apply(ZSC)
        for ii,biom in enumerate(df.columns): #biom is first column in sheet
            if ii >= num_columns:
                break
            vl = None
            if threshs:
                iii = temp * df.shape[1] + ii

                #print(biom, list(threshs.keys())[iii])
                vl = list(threshs.values())[iii]
                #not sure if dfs are sorted in same order as threshs, vl may match the biomarker but not subcat

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

                if i <= ncols_to_trim:
                    cor = df.loc[:,biom] * df.loc[:,bm] #series of multiples
                    #print(cor,biom,bm)
                    perc = np.percentile(cor,[percentile])[0]
                    #print(perc,percentile,'th percentile')
                    key = cor <= perc
                    #print(key.sum(), 'key sum')
                    #print(df.loc[:,bm].min(),' abc')
                    sheet[0:df.shape[0],i] = df.loc[:,bm].copy().where(cor <= perc,df.loc[:,bm].min())
                else:
                    sheet[0:df.shape[0],i] = df.loc[:,bm].copy()
                if i == 1:
                    scatterplot(sheet,0,1,biom,bm,vl=vl)
                i += 1
                #print(cors,mind)
                cors.pop(mind)
                bioms.pop(mind)
            data.append(sheet)
            #print(sheet)
            #print(len(data),biom,temp)
            #input()
    return(torch.tensor(data,dtype=torch.float32).permute(0, 2, 1))










# Example usage
if __name__ == "__main__":

    # Example parameters
    print('try to predict fraction of cells above threshold')

    num_columns = 2000 #11
    feature_dim = 100 #20000
    num_heads = 2  #feature_dim must be divisible by num_heads
    testDFs = 1

    DFs,threshs,zinfo = load(r'C:\Users\youm\Desktop\src\transformernet_training') #one DF per slide_scene
    batch_size = len(DFs)
    if batch_size < testDFs:
        testDFs = batch_size


    '''
    # Placeholder data and targets
    data = torch.randn(batch_size, num_columns, feature_dim)
    targets = torch.randn(batch_size)  # Random target thresholds for demonstration

    # Normalize targets
    targets = (targets - targets.mean()) / torch.sqrt(targets.var())
    '''

    # Initialize model
    model = ThresholdPredictor(feature_dim, num_heads)

    if TRAIN:
        data = makeData(DFs[:-testDFs],num_columns,feature_dim, percentile = 99, threshs = threshs)
        n_targets = data.size()[0] #one target for each biomarker for each slide/scene/etc (based on global variable)

        targets = torch.tensor(list(threshs.values())[:n_targets],dtype=torch.float32)
        #these targets are global zscores. model then predicts local scores. Not valid.
        #actually thresh is for each slide_scene
        #print(data,data.size(),'data')
        #print(DFs,'DFs')
        #print(threshs,'threshs')
        #input()
        #print(targets,'targets')
        #print(zinfo,'zinfo')

        print("Model Initialized!")
        # Train model
        predicted_thresholds, attention_weights = train_model(model, data, targets, max_epochs=MAXREPS, lr=LR, log_interval=int(MAXREPS/NFIGS))
        print(attention_weights.size())
        print(round((time.time()-STIME)/60,3),'minutes elapsed\n\n')
        torch.save(model.state_dict(), 'model_weights'+str(feature_dim)+'x'+str(num_columns)+'.pth')
    else: #num_columns and feature_dim must be the same as model was trained on- re-write to set these based on the model when loading. save relevant info in filename.
        model.load_state_dict(torch.load('model_weights'+str(feature_dim)+'x'+str(num_columns)+'.pth', weights_only=True)) #weights only: necessary?
        model.eval() #necessary?

    #test
    data = makeData(DFs[-testDFs:],num_columns,feature_dim)
    ziv, thv, zik, thk = list(zinfo.values())[-testDFs * num_columns:],list(threshs.values())[-testDFs * num_columns:],list(zinfo.keys())[-testDFs * num_columns:],list(threshs.keys())[-testDFs * num_columns:]
    #print(ziv,zik,'\n', thv,  thk,'\n')
    #print(data.size(),'test data size')
    predicted_thresholds, attention_weights = model(data)
    print(predicted_thresholds)
    for i in range(predicted_thresholds.size()[0]):
        print(predicted_thresholds[i])
        print(ziv[i],zik[i],'\n', thv[i],  thk[i],'\n')
    print(predicted_thresholds.size()[0],print(len(thv)))
    show(thv,predicted_thresholds.detach())



