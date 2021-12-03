import json
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from src.utils import load_data

with open('setup.json') as setupfile:
  setup = json.load(setupfile)

# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | Dataset class             |
# |___________________________|
class ABSADataset(Dataset):
  def __init__(self, df, tokenizer, max_len=setup['max_sen_length'], test_set = False):
    self.texts = df['text']
    self.aspects = df['aspect']
    self.test_set = test_set
    
    if self.test_set:
        self.targets = None
    else:
        self.targets = df['label']
    
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.aspects)

  def __getitem__(self, idx):

    # convert indexes, tensor->list
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    # define the aspect and text item
    text = (str(self.texts[idx]))
    aspect = str(self.aspects[idx])


    # pair the aspect and text for pair-encoding
    pairs = [text, aspect]
    
    # encode the feature pair
    encoded = self.tokenizer.encode_plus(pairs,
                                    add_special_tokens=True,
                                    padding='max_length', 
                                    max_length=setup['max_sen_length'], 
                                    return_attention_mask=True,
                                    return_tensors='pt',
                                    truncation=True)
    
    if not self.test_set:
        return {
            'input_ids' : encoded['input_ids'],
            'attention_mask' : encoded['attention_mask']
        }

    # define the labels => if not the test set
    target = self.targets[idx]

    return {
        'label' : target,
        'input_ids' : encoded['input_ids'],
        'attention_mask' : encoded['attention_mask'] 
    }



# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | Data Module - Lightning   |
# |___________________________|
class ABSADataModule(pl.LightningDataModule):
  def __init__(self,  transformer_tokenizer, batch_size: int = setup['train_batch_size'], num_workers: int = setup['num_workers']):
    super().__init__()

    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transformer_tokenizer = transformer_tokenizer


  # prepare the dataset and split
  def prepare_data(self):
    # load data
    df_train, df_test = load_data()

    # Add any preprocessing (if required) here:

    # Split data into training and validation
    self.df_train, self.df_valid = train_test_split(df_train, test_size=0.25, stratify=df_train['label'])
    self.df_test = df_test

    self.df_train.reset_index(drop=True, inplace=True)
    self.df_valid.reset_index(drop=True, inplace=True)

  # get the dataset obbjects    
  def setup(self, stage=None):
    self.train_dataset = ABSADataset(self.df_train, tokenizer=self.transformer_tokenizer)
    self.val_dataset = ABSADataset(self.df_valid, tokenizer=self.transformer_tokenizer)
    self.test_dataset = ABSADataset(self.df_test, tokenizer=self.transformer_tokenizer)

  # get the dataloader objects - PyTorch
  def train_dataloader(self):
    self.self.train_dataset = ABSADataset(self.df_train, tokenizer=self.transformer_tokenizer)
    return DataLoader(self.train_dataset, shuffle = True, batch_size=self.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    self.val_dataset = ABSADataset(self.df_valid, tokenizer=self.transformer_tokenizer)
    return DataLoader(self.val_dataset, shuffle = True, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    self.test_dataset = ABSADataset(self.df_test, tokenizer=self.transformer_tokenizer)
    return DataLoader(self.test_dataset, shuffle = True, batch_size=self.batch_size, num_workers=self.num_workers)



'''
    # |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    # | For Debugging            |
    # |__________________________|
    # print(f' text: {text}')
    # print(f' aspect: {aspect}')
    # print(type(text))
    # print(type(aspect))
    '''

    # Paste this after encoding step for debugging in datasets
    