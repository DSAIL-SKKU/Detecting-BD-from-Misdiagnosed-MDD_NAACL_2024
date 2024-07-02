import os
import argparse
import numpy as np
import random

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

## 추가
import configparser
import warnings
warnings.filterwarnings('ignore')

from src.MTL_classification import MTL_Classification

def th_seed_everything(seed: int = 2023):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

class Arg:
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    fp16: bool = False  # Enable train on FP16
    task_num: int = 2        
def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything(config['random_seed'])
    th_seed_everything(config['random_seed'])
       
    model = MTL_Classification(args,config) 
    model.preprocess_dataframe()

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    print(":: Start Training ::")
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback],
        enable_checkpointing = False,
        max_epochs=config['epochs'],
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True,
        gpus=[config['gpu']] if torch.cuda.is_available() else None, 
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader())
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #setting
    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--epochs", type=int, default=50, help="epoch")
    parser.add_argument("--dropout", type=float, default=0.1,help="dropout probablity")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--lr_scheduler", type=str, default='exp')   
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--n_fold", type=int, default=1) 
    #task
    parser.add_argument("--bf", type=int, default=730) 
    parser.add_argument("--af", type=int, default=730) 
    
    parser.add_argument("--task", type=str, default="mtl") # mtl, aux, main 
    #parser.add_argument("--eval", type=str, default="reg") 
    
    parser.add_argument("--main_num", type=int, default=2) 
    parser.add_argument("--aux_num", type=int, default=1) 
    
    parser.add_argument("--main_loss", type=str, default="cross")
    parser.add_argument("--aux_loss", type=str, default="reg")
    
    parser.add_argument("--embed_type", type=str, default="sen") 
    parser.add_argument("--hidden", type=int, default=1024) 
    #save
    parser.add_argument("--result_save", type=str, default="0803_ours")
    parser.add_argument("--model_save", type=str, default="ours") 
       
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       

