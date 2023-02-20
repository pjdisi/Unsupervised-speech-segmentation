from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, HubertConfig
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import soundfile as sf
import pandas as pd
import csv
import glob
import os
from datetime import datetime
from torch import nn
import math
from scipy.signal import find_peaks
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################################################################################

class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        torch.manual_seed(0)
        nhead=1
        self.pos_emb = PositionalEmbedding(1024)
        self.trans1=nn.TransformerEncoderLayer(1024,nhead=nhead,dropout=0)
        # self.trans2=nn.TransformerEncoderLayer(768,nhead=nhead,dropout=0)
        # self.trans3=nn.TransformerEncoderLayer(768,nhead=nhead,dropout=0)
        self.ffnn = nn.Sequential(
                        nn.Linear(1024,1),
                        nn.Sigmoid())

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
        pe = torch.zeros(max_len, 1, d_model).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe=pe
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

    
    
########################################################################################################################################

date = "2023_02_20-02_14_12_PM"
LANGUAGE = "mandarin"

if LANGUAGE=="english":
    PATH_ZERO_ENGLISH = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/english"
    paths = list(glob.glob(PATH_ZERO_ENGLISH+'/**/*.wav', recursive=True))
    limits = pd.read_csv("limits_english.csv", sep=" ",names=["Id","start","end"],index_col=False)
elif LANGUAGE=="french":
    PATH_ZERO_FRENCH = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/french"
    paths = list(glob.glob(PATH_ZERO_FRENCH+'/**/*.wav', recursive=True))
    limits = pd.read_csv("limits_french.csv", sep=" ",names=["Id","start","end"],index_col=False)
elif LANGUAGE=="mandarin":
    PATH_ZERO_MANDARIN = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/mandarin"
    paths = list(glob.glob(PATH_ZERO_MANDARIN+'/**/*.wav', recursive=True))
    limits = pd.read_csv("limits_mandarin.csv", sep=" ",names=["Id","start","end"],index_col=False)


model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53').to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model.load_state_dict(torch.load(date+'/best_hubert_parameters_'+date+'.pt',map_location=device))
model = model.eval()

net = NET().to(device)
net.load_state_dict(torch.load(date+'/best_net_parameters_'+date+'.pt',map_location=device))
net = net.eval()

TDE = True
file_name = "dummy.csv"

if TDE:
    file_disc = open("disc.txt","w")

with open(file_name,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["clip_id","start","end"])
    
    for counter, filename in tqdm(enumerate(paths)):
        clip_id = os.path.basename(filename).replace(".wav","")
            
        clip_audio, clip_sample_rate = sf.read(filename)
        limits_clip = limits[limits.Id==clip_id]
        
        input_values = processor(clip_audio, sampling_rate=clip_sample_rate, return_tensors="pt").input_values
        for _,limit in limits_clip.iterrows():
            start_limit, end_limit = np.floor(limit.start*16000).astype(int), np.floor(limit.end*16000).astype(int)
            
            if (limit.end-limit.start)<40:
                outputs = model(input_values[:,start_limit:end_limit+1].cuda()).last_hidden_state
                preds = net(outputs)
                pred_indexes = find_peaks(preds[0,:,0].detach().cpu(),prominence=0.1,distance=None,height=None)[0]
                
                for i in range(len(pred_indexes)-1):
                    start = (pred_indexes[i]/50)+start_limit/16000
                    end = (pred_indexes[i+1]/50)+start_limit/16000
                    writer.writerow([clip_id,start,end])
                    
                    if TDE:
                        file_disc.write("Class {a}:\n{b} {c} {d}\n".format(a=counter,b=clip_id,c=start,d=end))
            else:
                length_trim = 40
                num_trims = np.ceil((limit.end-limit.start)/length_trim).astype(int)
                for k in range(num_trims):
                    start_trim, end_trim = start_limit+k*length_trim*16000, start_limit+(k+1)*length_trim*16000
                    if k==num_trims-1:
                        end_trim = end_limit
                    outputs = model(input_values[:,start_trim:end_trim+1].cuda()).last_hidden_state
                    preds = net(outputs)
                    pred_indexes = find_peaks(preds[0,:,0].detach().cpu(),prominence=0.1,distance=None,height=None)[0]
                
                    for i in range(len(pred_indexes)-1):
                        start = (pred_indexes[i]/50)+start_trim/16000
                        end = (pred_indexes[i+1]/50)+start_trim/16000
                        writer.writerow([clip_id,start,end])
                        
                        if TDE:
                            file_disc.write("Class {a}:\n{b} {c} {d}\n".format(a=counter,b=clip_id,c=start,d=end))


if TDE:
    file_disc.write("\n")
    file_disc.close()
    
    if LANGUAGE=="english":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt english . -m boundary token/type")
    elif LANGUAGE=="french":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt french . -m boundary token/type")
    elif LANGUAGE=="mandarin":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt mandarin . -m boundary token/type")
        
    file = open("token_type","r")
    file = file.read()
    print(file)
    file = open("boundary","r")
    file = file.read()
    print(file)
