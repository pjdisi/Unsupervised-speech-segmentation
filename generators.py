import torch
from torch import nn
import numpy as np
import glob
import os
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import pandas as pd
import os
import re
import random
import math
from utils_bis import *
import time
from scipy.signal import convolve


PATH_LIBRISPEECH_TRAIN = "/scratch1/data/raw_data/LibriSpeech/train-clean-100"
PATH_LIBRISPEECH_TRAIN_360 = "/scratch1/data/raw_data/LibriSpeech/train-clean-360"
PATH_LIBRISPEECH_TRAIN_OTHER = "/scratch1/data/raw_data/LibriSpeech/train-other-500"
PATH_LIBRISPEECH_VAL = "/scratch1/data/raw_data/LibriSpeech/dev-clean"

PATH_ZEROSPEECH_ENGLISH = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/english"
PATH_ZEROSPEECH_FRENCH = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/french"
PATH_ZEROSPEECH_MANDARIN = "/scratch1/projects/zerospeech/2017/challenge_data/datasets/train/mandarin"

def gaussian_filter1d(size,sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [(1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return np.array(gaussian_filter)
gaussian_shape = gaussian_filter1d(5,1.3)

# Generator for zerospeech
def data_generator_zero_bis(batch_size,data,model,processor,lang="french",gaussian=False,shuffle=True):
    if lang=="french":
        paths = list(glob.glob(PATH_ZEROSPEECH_FRENCH+'/**/*.wav', recursive=True))
        limits = pd.read_csv("limits_french.csv", sep=" ",names=["Id","start","end"],index_col=False)
        val_ids = ['M03_N','M04_N','M05_N','M05_O','M06_N','M06_O','M08_O','M09_N','M09_O']
    elif lang=="mandarin":
        paths = list(glob.glob(PATH_ZEROSPEECH_MANDARIN+'/**/*.wav', recursive=True))
        limits = pd.read_csv("limits_mandarin.csv", sep=" ",names=["Id","start","end"],index_col=False)
        val_ids = ["A33","D08"]
    elif lang=="english":
        paths = list(glob.glob(PATH_ZEROSPEECH_ENGLISH+'/**/*.wav', recursive=True))
        limits = pd.read_csv("limits_english.csv", sep=" ",names=["Id","start","end"],index_col=False)
        val_ids = ['s2628','s4018','s5092','s0580','s5723','s5092']

    id_clip_dict = {}
    print("Preparing clips ...")
    for filename in tqdm(paths):
        clip_id = os.path.basename(filename).replace(".wav","")
        if clip_id in val_ids:
            continue
        clip_audio, clip_sample_rate = sf.read(filename)
        input_values = processor(clip_audio, sampling_rate=clip_sample_rate, return_tensors="pt").input_values
        id_clip_dict[clip_id] = input_values
    
    limits = limits[limits.Id.isin(val_ids)==False]
    limits = limits.sample(frac=1).reset_index(drop=True)
    
    number = 0
    for _,limit in limits.iterrows(): 
        o = np.min([batch_size-1,number])
        batch_iteration = np.min([o,number%batch_size])

        if number%batch_size==0:
            batch_outputs = torch.zeros((1,batch_size,num_vectors,1024))
            batch_labels = torch.zeros((1,batch_size,num_vectors,1))
            batch_padded = torch.ones_like(batch_labels)

        start, end = np.floor(limit.start*16000).astype(int), np.floor(limit.end*16000).astype(int)
        clip_id = limit.Id
        
        if limit.Id not in id_clip_dict.keys():
            print(limit.Id," not in dataset")
            continue

        indexes = get_boundary_indexes(data,clip_id,w2v_sample_rate)
        
        if ((end-start)/16000<25):
            outputs = model(id_clip_dict[clip_id][:,start:end+1].cuda()).last_hidden_state
            
            indexes_limit = indexes.clip((np.floor(limit.start*50)).astype(int), (np.floor(limit.start*50)).astype(int)+outputs.shape[1]-1)
            indexes_limit = indexes_limit - np.floor(limit.start*50).astype(int)
        else:
            print("VAD with length",(limit.end-limit.start),"ignored")
            continue
        labels = torch.zeros((1,1,outputs.shape[1],1)).cuda()
        outputs = torch.unsqueeze(outputs,0)

        if type(indexes)!=type(None):
            try:
                labels[0,0,indexes_limit,0] = 1.0
            except:
                continue
        else:
            continue

        outputs, labels, diff_lengths = get_clean_ol(outputs,labels)

        if gaussian:
            labels = labels.cpu().numpy()[0,0,:,0]
            labels = convolve(gaussian_shape,labels,"full","auto")[2:-2]
            labels /= np.max(gaussian_shape)
            labels = np.clip(labels,0,1)
            labels = torch.tensor(labels).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()

        batch_outputs[0,batch_iteration,:,:] = outputs[0,0,:,:]
        batch_labels[0,batch_iteration,:,:] = labels[0,0,:,:]

        if diff_lengths>0:
            batch_padded[0,batch_iteration,diff_lengths-1:,:] = 0
#         if type(indexes) == type(None):
#             batch_padded[0,batch_iteration,:,:] = 0
        number+=1

        if batch_iteration==batch_size-1:
            yield torch.Tensor(batch_outputs[0,:,:,:]),torch.Tensor(batch_labels[0,:,:,:]), torch.Tensor(batch_padded[0,:,:,:])

# Generator for Librispeech
def data_generator(batch_size,data,model,processor,gaussian=False,shuffle=True):
    paths = list(glob.glob(PATH_LIBRISPEECH_TRAIN+'/**/*.flac', recursive=True))
    paths += list(glob.glob(PATH_LIBRISPEECH_TRAIN_360+'/**/*.flac', recursive=True))
    paths += list(glob.glob(PATH_LIBRISPEECH_TRAIN_OTHER+'/**/*.flac', recursive=True))
    if shuffle:
        random.shuffle(paths)
    number = 0
    
    for filename in paths:
        o = np.min([batch_size-1,number])
        batch_iteration = np.min([o,number%batch_size])

        if number%batch_size==0:
            batch_outputs = torch.zeros((1,batch_size,num_vectors,768))
            batch_labels = torch.zeros((1,batch_size,num_vectors,1))
            batch_padded = torch.ones_like(batch_labels)

        clip_id = os.path.basename(filename).replace(".flac","")
        clip_audio, clip_sample_rate = sf.read(filename)

        indexes = get_boundary_indexes(data,clip_id,w2v_sample_rate)
        input_values = processor(clip_audio, sampling_rate=clip_sample_rate, return_tensors="pt").input_values.cuda()

        outputs = model(input_values).last_hidden_state
#         outputs = outputs.detach()

        labels = torch.zeros((1,1,outputs.shape[1],1)).cuda()
#         outputs = np.expand_dims(outputs,axis=0)
        outputs = torch.unsqueeze(outputs,0)
        if type(indexes)!=type(None):
            labels[0,0,indexes,0] = 1.0
        else:
            continue

        outputs, labels, diff_lengths = get_clean_ol(outputs,labels)

        if gaussian:
            labels = labels.cpu().numpy()[0,0,:,0]
            labels = convolve(gaussian_shape,labels,"full","auto")[2:-2]
            labels /= np.max(gaussian_shape)
            labels = np.clip(labels,0,1)
            labels = torch.tensor(labels).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()

        batch_outputs[0,batch_iteration,:,:] = outputs[0,0,:,:]
        batch_labels[0,batch_iteration,:,:] = labels[0,0,:,:]
        
        if diff_lengths>0:
            batch_padded[0,batch_iteration,diff_lengths-1:,:] = 0
        number+=1

        if batch_iteration==batch_size-1:
            yield torch.Tensor(batch_outputs[0,:,:,:]),torch.Tensor(batch_labels[0,:,:,:]), torch.Tensor(batch_padded[0,:,:,:])