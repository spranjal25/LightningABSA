import os
import json
import pandas as pd
import numpy as np


import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from sklearn.metrics import accuracy_score
import transformers
from argparse import ArgumentParser
from models.model import ABSASentimentClassifier

with open('setup.json') as setupfile:
    setup = setupfile

def load_data(setup):
    train = pd.read_csv(setup['file_path_train'])
    test = pd.read_csv(setup['file_path_test'])
    return train, test

def get_tokenizer(config=setup):
    return transformers.AutoTokenizer.from_pretrained(config['model_name'])


def Get_Class_weights(data):
    inverse_weights = np.array(data['label'].value_counts().sort_index())
    weights = np.sum(inverse_weights) / inverse_weights
    return weights

def get_training_args(setup=setup):
    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | TRAINING ARGUMENTS                |
    | these are project-wide arguments  |
    |___________________________________|
    '''
    root_dir = os.getcwd()
    print(root_dir)
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)

    # each LightningModule defines arguments relevant to it
    parser = ABSASentimentClassifier.add_model_specific_args(parent_parser, root_dir)

    assert torch.cuda.is_available(), "GPU(s) not present, to train on CPU, comment this line and set GPU=0 in parser.set_defualts"

    parser.set_defaults(
            #profiler='simple',
            deterministic=True,
            max_epochs=setup['epochs'],
            gpus=1,
            # weights=class_weights 
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            distributed_backend=None,
            fast_dev_run=False,
            model_load=False,
            model_name='lightning_logs/version_0/checkpoints/epoch=1-step=2343.ckpt'
        )

    args, extra = parser.parse_known_args()


    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | 2 CALLBACKS OF THE MODEL  |
    |___________________________|
    '''
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=setup['min_delta'],
        patience=setup['patience'],
        verbose=True,
        mode='min',
        strict=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        #dirpath='my/path/',
        filename='absa-classfiy-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

    trainer = Trainer.from_argparse_args(args,
        callbacks=[early_stop,lr_monitor, checkpoint_callback]
        ) 

    return extra, args, checkpoint_callback, early_stop, lr_monitor