import numpy as np
import pandas as pd
from datetime import datetime
from pprint import pprint
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score

def evaluation(config, task_type, loss_type,y_pred,y_true,_id):

    bf = config['bf']
    af = config['af']
    
    pred_dict = {}
    pred_dict['y_pred']= y_pred
    pred_dict['y_true']= y_true
 
    if task_type == 'main':
        val_score = classification_report(y_true, y_pred, output_dict=True) 
        pprint(val_score)
        val_df = pd.DataFrame.from_dict(val_score).T.reset_index()
        val_df = val_df.rename(columns = {'index':'category'})
        val_df['model'] = config['model_save']
        val_df['test_size'] = len(y_pred)
        val_df['n_fold'] = config['n_fold']
        val_df['task'] = config['task']
        val_df['bf'] = bf
        val_df['af'] = af
        
        val_df['config'] = str(config)
        
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred, squared=True)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mdae = median_absolute_error(y_true, y_pred)
        r_squared = r2_score(y_true, y_pred)
        
        val_df = { 
            'model' : [config['model_save']],
            'test_size': [len(y_pred)],
            'n_fold': [config['n_fold']],
            'task': [config['task']],
            'bf': [bf],
            'af': [af],
            'loss': [config[f'{task_type}_loss']],
            'mae': [mae],
            'rmse': [rmse],
            'mape': [mape],
            'mse': [mse],
            'mdae': [mdae],
            'r_squared': [r_squared],
            'config':[str(config)]}
        pprint(val_df)
    
    
    pred_dict = { 
        'model' : [config['model_save']]*len(y_pred),
        '_id' : _id,
        'test_size':  [len(y_pred)]*len(y_pred),
        'n_fold': [config['n_fold']]*len(y_pred),
        'bf': [bf]*len(y_pred),
        'af': [af]*len(y_pred),
        'loss': [config[f'{task_type}_loss']]*len(y_pred),
        'config':[str(config)]*len(y_pred),
        'true' : list(y_true),
        'pred' : list(y_pred),
    }

    # Result Save
    result_save = config['result_save']
    model_save = config['model_save']
    save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
    save_path = f"/home/dsail/daeun/dep2bd/model/_Result/{result_save}_{task_type}/"
    Path(f"{save_path}/pred").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(val_df).to_csv(f'{save_path}{save_time}_{model_save}_{task_type}.csv',index=False)  
    pd.DataFrame(pred_dict).to_csv(f'{save_path}pred/{save_time}_{model_save}_{bf}_{af}{task_type}_pred.csv',index=False)