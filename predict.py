import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from train import ABSASentimentClassifier
from src.utils import load_data, get_training_args, get_tokenizer

from src.dataloader import ABSADataset

import json

with open("setup.json") as setupfile:
    setup = json.load(setupfile)

def load_model_from_checkpoints(PATH:str):
    model = ABSASentimentClassifier.load_from_checkpoint(PATH)
    return model

def predict(PATH_TO_TEST, config=setup):
    _, test = load_data(config)

    model = load_model_from_checkpoints(config['checkpoint_path'])

    _, args, early_stop, checkpoint_callback, lr_monitor = get_training_args(config)
    
    transformer_tokenizer = get_tokenizer(config) 

    testset = ABSADataset(test, tokenizer=transformer_tokenizer, test_set=True)

    testLoader = DataLoader(testset, batch_size=setup['test_batch_size'])

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | 3 INIT Trainer   |
    |__________________|
    '''
    trainer = Trainer.from_argparse_args(
        args,callbacks=[early_stop,lr_monitor, checkpoint_callback]
    )

    preds = trainer.predict(model, testLoader)

    preds_final = []

    for pred_list in preds:
        for pred in pred_list:
            label = torch.argmax(pred)
            preds_final.append(label.cpu().detach().numpy().tolist())

    test['labels'] = preds_final
    test.to_csv(os.path.join(os.getcwd(), "data", "results", "test.csv"))



    

