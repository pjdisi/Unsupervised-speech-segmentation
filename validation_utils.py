import torch
from torch import nn
import numpy as np
import glob
import os
import soundfile as sf
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.spatial.distance import dice
import scipy.stats
import pandas as pd
from tqdm import tqdm
import pandas as pd
import os
import re
import random
import math
from utils_bis import *
from scipy.signal import find_peaks

# This function writes the predictions in a file that is readable for TDE
def write_tde(preds,clip_ids,starts,height=None,distance=None):
    file_disc = open("disc.txt","w")
    counter_vad = 0
    counter_speaker = 0
    clip_id_prev = clip_ids[0]
    mem = 0.0
    
    for clip_id,voice_clip_pred,mem in tqdm(zip(clip_ids,preds,starts)):
        
        if clip_id!=clip_id_prev:
            counter_speaker+=1
            
        clip_id_prev = clip_id
            
        # voice_clip_pred = (voice_clip_pred>0.5).astype(int)[0,:,:]
        voice_clip_pred_ = voice_clip_pred[0,:,0]
        
        starts, ends = discovery_policy(voice_clip_pred_,height=height,distance=distance)

        for start,end in zip(starts,ends):
            file_disc.write("Class {a}:\n{b} {c} {d}\n".format(a=counter_speaker,b=clip_id,c=start+mem,d=end+mem))
                     
    file_disc.write("\n")
    file_disc.close()
    return

# This function returns the start and end times for word segmentation predictions in a clip.
def discovery_policy(voice_clip_pred,height=None,distance=None):
    starts, ends = [], []
    indexes = find_peaks(voice_clip_pred,prominence=0.1,height=height,distance=distance)[0]
    for i in range(len(indexes)-1):
        starts.append(indexes[i]/50)
        ends.append(indexes[i+1]/50)
    return starts, ends

# This function runs TDE
def get_f1_tdev(preds,clip_ids,starts,height=None,distance=None,lang="english"):
    write_tde(preds,clip_ids,starts,height=height,distance=distance)
    if lang=="english":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt english_val . -m boundary token/type")   ## ENGLISH
    elif lang=="french":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt french_val . -m boundary token/type")   ## FRENCH
    elif lang=="mandarin":
        os.system("python /home/pdiegosimon/pablo/new_tdev2/tde/eval.py disc.txt mandarin_val . -m boundary token/type")   ## MANDARIN
    
    file = open("token_type","r")
    file = file.read()
    print(file)
    file = open("boundary","r")
    file = file.read()
    print(file)
    return

# This function calculates plain F1 metric
def get_f1_metric(preds,labels):
    final_preds = []
    final_labels = []
    for pred_clip,label_clip in zip(preds,labels):
        final_preds += list((pred_clip[0,:,0]>0.5).astype(int))
        final_labels += list(label_clip)
    f1_metric = f1_score(final_preds,final_labels)
    return f1_metric

# This function predicts over the voice clips in a corpus
def predict_val_finetune(net,model,processor,libri=False,lang="french"):
    clip_ids = []
    preds = []
    starts = []
    if libri:
        PATH_LIBRISPEECH_DEV = "/scratch1/data/raw_data/LibriSpeech/dev-clean"
        paths = list(glob.glob(PATH_LIBRISPEECH_DEV+'/**/*.flac', recursive=True))
        valid_ids = pd.read_csv("/home/pdiegosimon/pablo/new_tdev2/tde/share/dev-clean.wrd",header=None, sep=' ').iloc[:, 0].values.tolist()
        new_tdev2/tde/share/english.wrd
    else:
        if lang=="french":
            PATH_LIBRISPEECH_DEV = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/french"
            valid_ids = pd.read_csv("/home/pdiegosimon/pablo/new_tdev2/tde/share/french.wrd",header=None, sep=' ').iloc[:, 0].values.tolist()
            val_ids = ['M03_N','M04_N','M05_N','M05_O','M06_N','M06_O','M08_O','M09_N','M09_O']
            limits = pd.read_csv("limits_french.csv", sep=" ",names=["Id","start","end"],index_col=False)
            limits = limits[limits.Id.isin(val_ids)]
        elif lang=="mandarin":
            PATH_LIBRISPEECH_DEV = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/mandarin"
            valid_ids = pd.read_csv("/home/pdiegosimon/pablo/new_tdev2/tde/share/mandarin.wrd",header=None, sep=' ').iloc[:, 0].values.tolist()
            val_ids = ["A33","D08"]
            limits = pd.read_csv("limits_mandarin.csv", sep=" ",names=["Id","start","end"],index_col=False)
            limits = limits[limits.Id.isin(val_ids)]  
        elif lang=="english":
            PATH_LIBRISPEECH_DEV = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/english"
            valid_ids = pd.read_csv("/home/pdiegosimon/pablo/new_tdev2/tde/share/english.wrd",header=None, sep=' ').iloc[:, 0].values.tolist()
            val_ids = ['s2628','s4018','s5092','s0580','s5723','s5092']
            limits = pd.read_csv("limits_english.csv", sep=" ",names=["Id","start","end"],index_col=False)
            limits = limits[limits.Id.isin(val_ids)]  
    
    for clip_id in val_ids:
        path = PATH_LIBRISPEECH_DEV+"/"+clip_id+".wav"
        clip_audio, clip_sample_rate = sf.read(path)
        limits_clip = limits[limits.Id==clip_id]
        
        if clip_id in valid_ids:
            input_values = processor(clip_audio, sampling_rate=clip_sample_rate, return_tensors="pt").input_values
            for _,limit in limits_clip.iterrows():
                index_start = np.floor(limit.start*16000).astype(int)
                index_end = np.floor(limit.end*16000).astype(int)
                if (limit.end-limit.start)<40:
                    voice_clip_hubert = model(input_values[:,index_start:index_end+1].cuda()).last_hidden_state
                    net_output = net(torch.Tensor(voice_clip_hubert).cuda())

                    preds.append(net_output.cpu().detach().numpy())
                    clip_ids.append(clip_id)
                    starts.append(limit.start)
                else:
                    length_trim = 15
                    num_trims = np.ceil((limit.end-limit.start)/length_trim).astype(int)
                    for k in range(num_trims):
                        start_trim, end_trim = index_start+k*length_trim*16000, index_start+(k+1)*length_trim*16000
                        if k==num_trims-1:
                            end_trim = index_end
                        voice_clip_hubert = model(input_values[:,start_trim:end_trim+1].cuda()).last_hidden_state
                        net_output = net(torch.Tensor(voice_clip_hubert).cuda())
                        
                        preds.append(net_output.cpu().detach().numpy())
                        clip_ids.append(clip_id)
                        starts.append(limit.start+k*15)                  
    return preds, clip_ids, starts

# This function gets the boundaries for the calculation of plain F1 metric
def get_val_labels_bis(data,clip_ids,preds,starts):
    all_labels = []
    for pred,clip_id,start in zip(preds,clip_ids,starts):
        start_index = np.floor(start*50).astype(int)
        size = pred.shape[1]
        labels = np.zeros((size,))
        indexes = get_boundary_indexes(data,clip_id,w2v_sample_rate)
        indexes = indexes.clip(start_index,start_index+size-1)-start_index
        labels[indexes] = 1.0
        all_labels.append(labels)
    return all_labels

# This function predicts the peaks to calculate plain F1
def get_peak_preds(preds,prominence_candidate,height_candidate,distance_candidate):
    preds_peak = []
    for i,pred in enumerate(preds):
        peak_ind = find_peaks(pred[0,:,0],prominence=prominence_candidate,height=height_candidate,distance=distance_candidate)[0]
        empty = np.zeros((1,pred.shape[1],1))
        empty[0,peak_ind,0] = 1
        empty[0,peak_ind+1,0] = 1
        empty[0,peak_ind-1,0] = 1
        preds_peak.append(empty)
    return preds_peak

# This function searches for different hyperparams for the peak detection method
def search_f1_hyper_params(preds,labels):
    f1_metric_candidates = []
    best_f1 =  0
    best_height = None
    best_distance = None
    best_prominence = None
    
    for height_candidate in np.linspace(0,1,20):
        for distance_candidate in np.arange(1,5):
            # preds_f1_candidate = get_peak_preds(preds,0.1,height_candidate,distance_candidate)
            preds_f1_candidate = get_peak_preds(preds,0.1,None,None) # BEFORE THIS WAS ALL NONE
            f1_candidate = get_f1_metric(preds_f1_candidate,labels)
            f1_metric_candidates.append(f1_candidate)
            if f1_candidate > best_f1:
                best_height = height_candidate
                best_distance = distance_candidate

                best_f1 = f1_candidate
                
    f1_metric_candidates = np.array(f1_metric_candidates)
    f1_metric = np.max(f1_metric_candidates)
    
    return f1_metric, best_height, best_distance, best_prominence


# These fucntions read the output TDE files and parses scores from them.
def get_boundary_tdev_float():
    f = open("boundary","r").readlines()[3]
    index1 = f.find(":")
    index2 = f.find("\n")
    f1_tdev = float(f[index1+2:index2])
    return f1_tdev

def get_token_tdev_float():
    f = open("token_type","r").readlines()[3]
    index1 = f.find(":")
    index2 = f.find("\n")
    f1_tdev = float(f[index1+2:index2])
    return f1_tdev

def get_num_disc():
    f = open("disc.txt","r")
    num_lines = len(f.readlines())
    num_disc = (num_lines-1)/2
    return num_disc