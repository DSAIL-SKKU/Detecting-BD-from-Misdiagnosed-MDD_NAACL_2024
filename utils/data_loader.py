import torch
from torch.utils.data import Dataset
import torch.nn as nn
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
import configparser

file_path = configparser.ConfigParser()
file_path.read('./utils/path.ini')
    
def data_load(args, config):

    main_y = 'trans_' + str(config['af']) + '_y' #'shift_label'#
    aux_y = 'mood_label'
    embed = config['embed_type']

    if embed in ['go','long','sen','emo']:
        data_path = file_path['data_path'][str(config['bf'])]
    else:
        data_path = file_path['data_path'][str(config['bf'])]
    
    df = pd.read_json(data_path)
    df[aux_y] = [[l-3+ 0.1 for l in labels] for labels in df['mood_label']] 
    
    # data split
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['random_seed'])        
    for i, (train_idxs, test_idxs) in enumerate(cv.split(df, df[main_y], df['author'])):
        if i == config['n_fold']:
            print("TRAIN:", len(train_idxs), " TEST: ", len(test_idxs))
            break   
                            
    df_train = df.iloc[train_idxs].sort_values('window_id')    
    df_test = df.iloc[test_idxs].sort_values('window_id')  
        
    # oversampling
    ros = RandomOverSampler(random_state=2023)
    df_train, y_res = ros.fit_resample(df_train, df_train[main_y].tolist())
    print(f'oversampling : # of train:{len(df_train)}, val:0, test:{len(df_test)}')
            
    train_data = RedditDataset(
        df_train['window_id'].values,
        df_train['post_id'].values,
        df_train['created_utc'].values,
        df_train[f'{embed}_enc'].values,
        df_train[aux_y].values,
        df_train[main_y].values,
        df_train['m_change_1st'].values,
    )
    test_data = RedditDataset(
        df_test['window_id'].values,
        df_test['post_id'].values,
        df_test['created_utc'].values,
        df_test[f'{embed}_enc'].values,
        df_test[aux_y].values,
        df_test[main_y].values,
        df_test['m_change_1st'].values,
    )
                        
    return train_data,test_data


class RedditDataset(Dataset):
    def __init__(self, window_id, post_id, timestamp, emo_enc, mood_label,trans_label,change_label,days=50):
        super().__init__()
        
        self.window_id = window_id
        self.trans_label = trans_label
        
        self.post_id = post_id
        self.timestamp = timestamp
        self.emo_enc = emo_enc
        
        self.mood_label = mood_label
        self.change_label = change_label
        
        self.days = days

    def __len__(self):
        return len(self.window_id)

    def __getitem__(self, item):
        
        window_id = torch.tensor(self.window_id[item], dtype=torch.long)
        trans_label = torch.tensor(self.trans_label[item], dtype=torch.long)
        
        if self.days > len(self.emo_enc[item]):
            post_id = torch.tensor(self.post_id[item], dtype=torch.long)
            timestamp = torch.tensor(get_timestamp(self.timestamp[item]), dtype=torch.float32)
            emo_enc = torch.tensor(self.emo_enc[item], dtype=torch.float32)
            mood_label = torch.tensor(self.mood_label[item], dtype=torch.float32)
            change_label = torch.tensor(self.change_label[item], dtype=torch.float32)
            
        else:
            post_id = torch.tensor(self.post_id[item][:self.days], dtype=torch.long)
            timestamp = torch.tensor(get_timestamp(self.timestamp[item][:self.days]), dtype=torch.float32)
            emo_enc = torch.tensor(self.emo_enc[item][:self.days], dtype=torch.float32)
            mood_label = torch.tensor(self.mood_label[item][:self.days], dtype=torch.float32)
            change_label = torch.tensor(self.change_label[item][:self.days], dtype=torch.float32)
            
        return [window_id, post_id, timestamp,emo_enc, mood_label,trans_label,change_label]


def pad_collate_reddit(batch): 
    
    window_id =torch.tensor([item[0] for item in batch])
    post_id  =[item[1] for item in batch]
    timestamp =[item[2] for item in batch]
    emo_enc =[item[3] for item in batch]
    
    mood_label  =[item[4] for item in batch]
    trans_label=torch.tensor([item[5] for item in batch])
    change_label  =[item[6] for item in batch]
    
    p_num = torch.tensor([len(x) for x in emo_enc])
    
    post_id = nn.utils.rnn.pad_sequence(post_id, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)
    emo_enc = nn.utils.rnn.pad_sequence(emo_enc, batch_first=True, padding_value=0)
    
    mood_label = nn.utils.rnn.pad_sequence(mood_label, batch_first=True, padding_value=0)
    change_label = nn.utils.rnn.pad_sequence(change_label, batch_first=True, padding_value=0)

    return [window_id, post_id, timestamp, emo_enc,mood_label,trans_label,p_num,change_label]


def get_timestamp(x):
    def change_utc(x):
        try:
            x = str(datetime.fromtimestamp(int(x)/1000))
            return x
        except:
            return str(x)

    timestamp = [datetime.timestamp(datetime.strptime(change_utc(t),"%Y-%m-%d %H:%M:%S")) for t in x]
    timestamp = (timestamp[0] - np.array(timestamp))
    return timestamp