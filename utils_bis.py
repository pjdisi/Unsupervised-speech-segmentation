import torch
import torch.nn.functional as f
from torch import nn
import numpy as np
import glob
import os
import soundfile as sf
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import pandas as pd
import os
import re
import random
import math
import time
from matplotlib import pyplot as plt

w2v_sample_rate = 50
audio_sampling_rate = 16000
audio_length = 15
text_length = 200
num_vectors = w2v_sample_rate*audio_length

# From the segmentation of a voice clip, this function returns a list of indexes where boundaries are located. (50Hz frequency)
def get_boundary_indexes(data,clip_id,sample_rate):
    voice_clip = data.loc[data['Id'] == clip_id]
    ends = voice_clip['end'].to_numpy()
    starts = voice_clip['start'].to_numpy()

    if len(ends)==0:
        assert clip_id not in set(data["Id"])
        return None
    
    ends = np.insert(ends,0,0)

    if "word" not in voice_clip.columns:
        silences = voice_clip.loc[voice_clip['start']!=voice_clip.shift(1)["end"],["start","end"]].to_numpy()
    else:
        silences = voice_clip.loc[voice_clip['word']=="SIL",['start','end']].to_numpy()
    
    silences = np.floor(silences*sample_rate).astype(int)
    indexes = np.floor(ends*sample_rate).astype(int)
    
    indexes = np.sort(indexes)
    max_index = indexes[-1]

    indexes = np.append(indexes,indexes-1)
    indexes = np.append(indexes,indexes+1)

    indexes_ = np.floor(starts*sample_rate).astype(int)
    indexes = np.append(indexes,indexes_)
    indexes = np.append(indexes,indexes_-1)
    indexes = np.append(indexes,indexes_+1)
    
    indexes = np.clip(indexes,0,max_index)
    
    indexes = np.unique(indexes)
    indexes = np.clip(indexes,0,max_index-2)
    return indexes

# This function crops the outputs of Wav2Vec2 so they are 750 samples long. If the outputs are smaller, the function pads with zeros until length 750 is achieved.
def get_clean_ol(outputs,labels):
    if outputs.shape[2]>num_vectors:
        outputs = outputs[:,:,:num_vectors,:].cuda()
        labels = labels[:,:,:num_vectors,:].cuda()
        diff_lengths = 0

    elif outputs.shape[2]<=num_vectors:
        outputs = outputs.cuda()
        labels = labels.cuda()
        
        diff_lengths = num_vectors-outputs.shape[2]
        
        empty_mat = torch.zeros((1,1,diff_lengths,outputs.shape[-1])).cuda()
        outputs = torch.cat((outputs,empty_mat),dim=2)
    
        empty_vec = torch.zeros((1,1,diff_lengths,1)).cuda()
        labels = torch.cat((labels,empty_vec),dim=2)
    return outputs, labels, diff_lengths

# This function calculates the BCE loss only for those samples that have not been padded previously
def indexed_BCE(preds,labels,batch_padded,criterion):
    preds = preds.view(-1)
    labels = labels.view(-1)
    batch_padded = batch_padded.view(-1)

    indexes = torch.argwhere(batch_padded==1)

    preds_clean = preds[indexes]
    labels_clean = labels[indexes]

    batch_loss = criterion(preds_clean,labels_clean)
    return batch_loss

# This function calculates the BCE loss only for those samples that have not been padded previously. Only those smaples amonng the 5th and 9th deciles of all the batch losses.
def indexed_BCE_trick(preds,labels,batch_padded,criterion):
    preds = preds.view(-1)
    labels = labels.view(-1)
    batch_padded = batch_padded.view(-1)

    indexes = torch.argwhere(batch_padded==1)

    preds_clean = preds[indexes]
    labels_clean = labels[indexes]

    loss = criterion(preds_clean,labels_clean)
    down, up = torch.quantile(loss,torch.tensor([0.5,0.9]).cuda(),interpolation='nearest')
    indexes = torch.logical_and(loss<up,loss>down)
    loss_final = torch.mean(loss[indexes])
    return loss_final    

# This function calculates the F1 score only for those samples that have not been padded previously.
def indexed_F1(preds,labels,batch_padded,fun):
    preds = preds.view(-1)
    labels = labels.view(-1)
    batch_padded = batch_padded.view(-1)

    indexes = torch.argwhere(batch_padded==1)
    preds_clean = preds[indexes].detach().cpu().numpy()
    labels_clean = labels[indexes].detach().cpu().numpy()

    preds_clean = (preds_clean>0.5).astype(int)
    try:
        batch_f1 = fun(labels_clean,preds_clean)
    except:
        batch_f1 = None
    return batch_f1

# Here we define the net that is on top of Wav2Vec2.0
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        torch.manual_seed(0)
        nhead=1
        self.pos_emb = PositionalEmbedding(1024).cuda()
        self.trans1=nn.TransformerEncoderLayer(1024,nhead=nhead,dropout=0).cuda()
        # self.trans2=nn.TransformerEncoderLayer(768,nhead=nhead,dropout=0).cuda()
        # self.trans3=nn.TransformerEncoderLayer(768,nhead=nhead,dropout=0).cuda()
        self.ffnn = nn.Sequential(
                        nn.Linear(1024,1),
                        nn.Sigmoid()).cuda()

    def forward(self, x):
        x=x.permute(1, 0, 2)
        x=self.pos_emb(x)
        x=self.trans1(x)
        # x=self.trans2(x)
        # x=self.trans3(x)
        x=x.transpose(0,1)
        x = self.ffnn(x)
        return x

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe=pe.cuda()
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]
            