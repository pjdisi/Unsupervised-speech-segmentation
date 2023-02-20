import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, HubertConfig
from generators import *
from validation_utils import *
import soundfile as sf
from torch import nn
import pandas as pd
import glob
import os
import sys
from datetime import datetime
from sklearn.metrics import f1_score, recall_score, precision_score

DATE_STRING = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
os.mkdir(DATE_STRING)

BATCH_SIZE = 6
LR = 0.0001
EPOCHS = 50
PATIENCE_NUM = 4
VAL = True
GET_LOSS_CURVE = True # Whether we want to get the training curves
GET_VAL = True # Whether we want to evaluate the model on the validation set every 25 batch iterations
LANGUAGE = "mandarin"

# We define the processor, the self-supervised model we use and the network on top
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").train().cuda()
net = NET().cuda()

# We activate training for those layers in the self-supervised model we want
for name, param in model.named_parameters():
    if param.requires_grad:
        if name.startswith("feature") or name=="masked_spec_embed":
            param.requires_grad = False


if LANGUAGE=="english":
    unsupervised_segmentation = pd.read_csv('english2017_dpparse', header = None, sep=" ", names=["Id","start","end"])
    data_val = pd.read_csv('english_val_dp.csv', header = None, sep=" ", names=["Id","start","end","word"],index_col=False)
elif LANGUAGE=="french":
    unsupervised_segmentation = pd.read_csv('french_dpparse', header = None, sep=" ", names=["Id","start","end"])
    data_val = pd.read_csv('french_val_dp.csv', header = None, sep=" ", names=["Id","start","end","word"],index_col=False)
elif LANGUAGE=="mandarin":
    unsupervised_segmentation = pd.read_csv('mandarin_dpparse', header = None, sep=" ", names=["Id","start","end"])
    data_val = pd.read_csv('mandarin_val_dp.csv', header = None, sep=" ", names=["Id","start","end","word"],index_col=False)

# If we want to loop, we have to take the previous predictions as unsupervised segmentation file
# unsupervised_segmentation = pd.read_csv('dummy.csv',header=0,sep=",",names=["Id","start","end"],index_col=False)


optimizer = optim.Adam(list(net.parameters())+list(model.parameters()), lr=LR)

criterion = nn.BCELoss()
criterion_reg = nn.MSELoss()
criterion_trick = torch.nn.BCELoss(reduction="none")

loss_history = []
f1_history = []
f1_val_history = []
f1_boundary_tdev_history = []
f1_token_tdev_history = []
best_boundary_tde = 0.0
best_f1_metric = 0.0
total_iter = 0

for epoch_num in range(EPOCHS):
    data_loader = data_generator_zero_bis(BATCH_SIZE,unsupervised_segmentation,model,processor,lang=LANGUAGE)
    counter = 0
    print("EPOCH ",epoch_num+1, " has started.")
    for outputs, labels, batch_padded  in data_loader:
        try:
            preds = net(outputs.cuda())
        except Exception as e:
            print("ITERATION FAILED")
            print(e)
            continue

        optimizer.zero_grad()

        batch_padded = batch_padded.cuda()
        loss = indexed_BCE_trick(preds,labels.cuda(),batch_padded,criterion_trick)

        batch_f1 = indexed_F1(preds,labels,batch_padded,f1_score)
        batch_recall = indexed_F1(preds,labels,batch_padded,recall_score)
        batch_precision = indexed_F1(preds,labels,batch_padded,precision_score)
        
        loss.backward()        
        optimizer.step()
            
        print("Loss:",loss.item()," F1 score:",batch_f1," Recall score:",batch_recall," Precision score:",batch_precision)

        loss_history.append(loss.cpu().item())
        f1_history.append(batch_f1)

        if counter%(3) == 0.0 and counter!=0 and GET_LOSS_CURVE:
            np.save(DATE_STRING+"/training_loss"+DATE_STRING+".npy",np.array(loss_history))
            np.save(DATE_STRING+"/training_F1"+DATE_STRING+".npy",np.array(f1_history))
        counter+=1
        total_iter+=1

        if VAL and total_iter%25==0 and counter!=0:
            print("ITERATION NUMBER: ",total_iter)
            model.eval()
            net.eval()

            preds, clip_ids, starts = predict_val_finetune(net,model,processor,libri=False,lang=LANGUAGE)
            labels = get_val_labels_bis(data_val,clip_ids,preds,starts)
                    
            model.train()
            net.train()
            
            f1_metric, height, distance, prominence = search_f1_hyper_params(preds,labels) # This is disabled, looping uselessly
            
            get_f1_tdev(preds,clip_ids,starts,height=height,distance=distance,lang=LANGUAGE)
            f1_boundary_tde = get_boundary_tdev_float()
            f1_token_tde = get_token_tdev_float()
            num_disc = get_num_disc()

            f1_boundary_tdev_history.append(f1_boundary_tde)
            f1_token_tdev_history.append(f1_token_tde)
            f1_val_history.append(f1_metric)
            
            if len(f1_val_history)>PATIENCE_NUM:
                patience =  ((np.array(f1_val_history[-PATIENCE_NUM:])-f1_val_history[-PATIENCE_NUM])<=0).astype(int)
                if sum(patience)==PATIENCE_NUM:
                    sys.exit()
                        
            if GET_VAL:
                torch.save(net.state_dict(), DATE_STRING+"/net_parameters_{}.pt".format(DATE_STRING))
                torch.save(model.state_dict(), DATE_STRING+"/hubert_parameters_{}.pt".format(DATE_STRING))
                if GET_LOSS_CURVE:
                    np.save(DATE_STRING+"/val_F1"+DATE_STRING+".npy",np.array(f1_val_history))
                    np.save(DATE_STRING+"/tde_boundary_F1"+DATE_STRING+".npy",np.array(f1_boundary_tdev_history))
                    np.save(DATE_STRING+"/tde_token_F1"+DATE_STRING+".npy",np.array(f1_token_tdev_history))
            print("VAL F1: ",f1_metric)
            print("HEIGHT:",height,"\nDISTANCE:",distance,"\nPROMINENCE:",prominence)
            print("BOUNDARY TDE F1: ",f1_boundary_tde)
            print("TOKEN TDE F1: ",f1_token_tde)
            print("NUM DISC:", num_disc)
            print("CODE: ", DATE_STRING,"\n")
            
            if (f1_metric>best_f1_metric) and GET_VAL:
                torch.save(net.state_dict(), DATE_STRING+"/best_net_parameters_{}.pt".format(DATE_STRING))
                # best_boundary_tde = f1_boundary_tde
                best_f1_metric = f1_metric
                with open("iter_number.txt","w") as f:
                    f.write("TOTAL ITER: "+str(total_iter))
                    f.write("\nHEIGHT: "+str(height))
                    f.write("\nDISTANCE: "+str(distance))
                    f.write("\nPROMINENCE: "+str(prominence))
                    f.close()

                torch.save(model.state_dict(), DATE_STRING+"/best_hubert_parameters_{}.pt".format(DATE_STRING))

