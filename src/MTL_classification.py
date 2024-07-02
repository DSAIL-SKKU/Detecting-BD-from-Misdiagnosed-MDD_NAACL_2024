import os
import pandas as pd
import numpy as np
import argparse

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule
from transformers import AdamW

from sklearn.model_selection import StratifiedGroupKFold
#from imblearn.over_sampling import RandomOverSampler

from src.attention import Attention
from src.transformerEncoder import TransformerEncoder
from src.position_embedding import SinusoidalPositionalEmbedding

import sys
sys.path.append('../')
from utils.loss import loss_function
from utils.evaluation import *
from utils.data_loader import data_load, pad_collate_reddit

class Arg:
    epochs: int = 50  # Max Epochs, BERT paper setting [3,4,5]
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    batch_size: int =32 
    max_post_num = 30
    task_num: int = 2

class MTL_Classification(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        self.args = args
        self.config = config
        
        # hist feat #추후 정리
        self.num_heads = 2
        self.layers = 4 
        self.attn_dropout = 0.1 
        self.relu_dropout = 0.1 
        self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.1
        self.attn_mask = False 
        
        self.hist_enc = TransformerEncoder(embed_dim=self.config['hidden'],
                                          num_heads=self.num_heads,
                                          layers=max(self.layers, -1),
                                          attn_dropout=self.attn_dropout,
                                          relu_dropout=self.relu_dropout,
                                          res_dropout=self.res_dropout,
                                          embed_dropout=self.embed_dropout,
                                          attn_mask=self.attn_mask)
        
        self.time_var = nn.Parameter(torch.randn((2)), requires_grad=True)
        
        self.embed_positions = SinusoidalPositionalEmbedding(self.config['hidden'])
        self.atten = Attention(self.config['gpu'],self.config['hidden'], batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        self.conv = nn.Conv1d(self.config['hidden'], self.config['hidden'], kernel_size=1, padding=0, bias=False)

        # suicide
        self.fc_1 = nn.Linear(self.config['hidden'], self.config['hidden'])
        self.fc_2 = nn.Linear(self.config['hidden'], self.config['main_num'])
        
        # aux
        self.m_decoder = nn.Linear(self.config['hidden'], self.config['aux_num'])
        self.c_decoder = nn.Linear(self.config['hidden'], self.config['aux_num'])
        
        # unweighted loss
        self.log_vars = nn.Parameter(torch.randn((self.args.task_num)))
        
    def forward(self,data):  
        window_id, post_id, timestamp, emo_enc, aux_y,main_y,p_num,change_label  = data 
        
        x = self.dropout(emo_enc)   
        
        x = nn.ReLU()(self.conv(x.transpose(1,2)).transpose(1,2))
        
        c_out = self.c_decoder(x) 
        b_out = self.m_decoder(x) #+ c_out
        
        post_id = nn.utils.rnn.pack_padded_sequence(post_id, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        
        aux_logits = nn.utils.rnn.pack_padded_sequence(b_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        aux_y = nn.utils.rnn.pack_padded_sequence(aux_y, p_num.cpu(), batch_first=True, enforce_sorted=False)[0]
        aux_loss = loss_function(aux_logits, aux_y, self.config['aux_loss'], self.config['aux_num'], 1.8)
        
        change_logits = nn.utils.rnn.pack_padded_sequence(c_out, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        change_label = nn.utils.rnn.pack_padded_sequence(change_label, p_num.cpu(), batch_first=True, enforce_sorted=False)[0] 
        change_loss = loss_function(change_logits, change_label, self.config['aux_loss'], self.config['aux_num'], 1.8)
        aux_loss = aux_loss + change_loss
        
        hist, attn_score = self.hist_enc(x+ self.embed_positions(x.transpose(0, 1)[:, :, 0]).transpose(0, 1))

#         # main
        timestamp = torch.exp(self.time_var[0]) *timestamp + self.time_var[0]
        timestamp = torch.sigmoid(timestamp+ self.time_var[1]) #.size()
        x = x+ x*timestamp.unsqueeze(-1)
        h, att_score = self.atten(x+hist, p_num.cpu())  # skip connect #+hist
        
        #reddit model        
        if h.dim() == 1:
            h = h.unsqueeze(0) 
        
        main_logits = self.fc_2(nn.Tanh()(self.fc_1(self.dropout(h))))
        main_loss = loss_function(main_logits, main_y, self.config['main_loss'], self.config['main_num'], 1.8)

        # multi task loss            
        s_prec = torch.exp(-self.log_vars[0])
        main_loss = s_prec*main_loss + self.log_vars[0]
        
        b_prec = torch.exp(-self.log_vars[1])
        aux_loss = b_prec*aux_loss + self.log_vars[1]
        
        total_loss =  main_loss  + aux_loss #+ (s_prec-b_prec)
        
#         aux_logits = main_logits
#         aux_y = main_y
#         post_id = window_id
        return total_loss, main_logits,aux_logits, aux_y,post_id,window_id,main_y
    

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])#W
        scheduler = ExponentialLR(optimizer, gamma=0.01)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        self.train_data, self.test_data = data_load(self.args, self.config)    

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config['batch_size'],
            collate_fn=pad_collate_reddit,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config['batch_size'],
            collate_fn=pad_collate_reddit,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        loss, main_logits,aux_logits, aux_y,post_id,window_id,main_y = self(batch)
        self.log("train_loss", loss)
            
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss, main_logits,aux_logits, aux_y,post_id,window_id,main_y = self(batch)

        aux_pred = list(aux_logits.cpu().numpy()) #.argmax(dim=-1)
        aux_true = list(aux_y.cpu().numpy())
        aux_id = list(post_id.cpu().numpy())
        
        #main
        main_pred = list(main_logits.argmax(dim=-1).cpu().numpy())
        main_true = list(main_y.cpu().numpy())
        main_id = list(window_id.cpu().numpy())


        return {
            'main_pred': main_pred,
            'main_true': main_true,
            'main_id': main_id,
            'aux_pred': aux_pred,
            'aux_true': aux_true,
            'aux_id': aux_id,
        }

    def test_epoch_end(self, outputs):
        for task_type in ['main']: #,'aux'
            print(f'{task_type} evaluation........')
            y_true = []
            y_pred = []
            _id = []
            for i in outputs:
                y_true += i[f'{task_type}_true']
                y_pred += i[f'{task_type}_pred']
                _id += i[f'{task_type}_id']

            y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
            y_true = np.asanyarray(y_true)
            _id = np.asanyarray(_id)

            evaluation(self.config, task_type, self.config[f'{task_type}_loss'], y_pred,y_true,_id)

