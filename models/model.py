import os

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
import transformers
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from argparse import ArgumentParser

# =============================================================================================================|

def Get_Model(model:str='bert'):
    if model == 'bert':
        return ABSAModel_Bert()

    elif model == 'distilbert':
        return 

# =============================================================================================================|

# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | distilBERT - Model  |
# |_____________________|
class ABSAModel_distilBert(torch.nn.Module):

  def __init__(self, setup, **kwargs):
    super(ABSAModel_Bert, self).__init__()
    
    self.num_labels=setup['num_labels'],

    # load AutoModel & Autoconfig
    self.distilbert = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    self.distilbert_config = transformers.AutoConfig.from_pretrained('distilbert-base-uncased')

    self.pre_classifier = torch.nn.Linear(self.distilbert_config.dim, self.distilbert_config.dim)

    self.classifier = torch.nn.Linear(self.distilbert_config.dim, self.num_labels)

    self.dropout = torch.nn.Dropout(self.distilbert_config.dropout)
    # print(f'Using Dropout = {self.bert.config.seq_classif_dropout}')

    self.relu = torch.nn.ReLU()

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | freeze the layers of Bert for training if needed so that |   
    | the embeddings of all layers of Bert are not changed     |
    |__________________________________________________________|
    '''
    for param in self.bert.parameters():
      param.requires_grad = False

  
  def forward(self, batch):
  #   print((batch['input_ids'].squeeze(1)).shape)
  #   print("*"*10)
  #   print(batch['input_ids'])
  #   print("*"*10)
    outputs = self.distilbert(input_ids=batch['input_ids'].squeeze(1), 
                        attention_mask=batch['attention_mask'])
    
    # output from last hidden layer
    hidden_state = outputs[0]  # (batch_size, seq_len, dim)

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | *output of [CLS] token                                   |
    |                                                          |
    | [CLS] token contains the pooled embeddings of the entire | 
    | Sequence, these are used for the classification.         |
    |__________________________________________________________|
    '''
    pooled_output = hidden_state[:, 0] # (batch_size, dim)

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sending the [CLS] token embeddings through Linear, ReLU  |
    | and Dropout layers                                       |
    |__________________________________________________________|
    '''
    pooled_output = self.pre_classifier(pooled_output)  # (batch_size, dim)
    pooled_output = self.relu(pooled_output)  # (batch_size, dim)
    pooled_output = self.dropout(pooled_output)  # (batch_size, dim)
    logits = self.classifier(pooled_output)  # (batch_size, num_labels)

    return logits

  def get_outputs(self, input_ids, attention_mask):
    outputs = self.distilbert(input_ids=input_ids, \
                        attention_mask=attention_mask)

    return outputs

# ==================================================================================================================================|

# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | BERT - Model      |
# |___________________|

class ABSAModel_Bert(torch.nn.Module):

  def __init__(self, config, **kwargs):
    super(ABSAModel_Bert, self).__init__()
    self.config = config
    self.num_labels=self.config['num_labels']
    self.bert = transformers.AutoModel.from_pretrained(self.config['model_name'])
    self.bert_config = transformers.AutoConfig.from_pretrained(self.config['model_name'])

    self.pre_classifier = torch.nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)

    self.classifier = torch.nn.Linear(self.bert_config.hidden_size, self.num_labels)

    self.dropout = torch.nn.Dropout(self.bert_config.hidden_dropout_prob)
    # print(f'Using Dropout = {self.bert.config.seq_classif_dropout}')

    self.relu = torch.nn.ReLU()

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | freeze the layers of Bert for training if needed so that |   
    | the embeddings of all layers of Bert are not changed     |
    |__________________________________________________________|
    '''
    for param in self.bert.parameters():
      param.requires_grad = False

  
  def forward(self, batch):
  #   print((batch['input_ids'].squeeze(1)).shape)
  #   print("*"*10)
  #   print(batch['input_ids'])
  #   print("*"*10)
    outputs = self.bert(input_ids=batch['input_ids'].squeeze(1), 
                        attention_mask=batch['attention_mask'])
    
    # output from last hidden layer
    hidden_state = outputs[0]  # (batch_size, seq_len, dim)

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | *output of [CLS] token                                   |
    |                                                          |
    | [CLS] token contains the pooled embeddings of the entire | 
    | Sequence, these are used for the classification.         |
    |__________________________________________________________|
    '''
    pooled_output = hidden_state[:, 0] # (batch_size, dim)

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | sending the [CLS] token embeddings through Linear, ReLU  |
    | and Dropout layers                                       |
    |__________________________________________________________|
    '''
    pooled_output = self.pre_classifier(pooled_output)  # (batch_size, dim)
    pooled_output = self.relu(pooled_output)  # (batch_size, dim)
    pooled_output = self.dropout(pooled_output)  # (batch_size, dim)
    logits = self.classifier(pooled_output)  # (batch_size, num_labels)

    return logits

  def get_outputs(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, \
                        attention_mask=attention_mask)

    return outputs


# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | Define the Pytorch Lightning Module Classifier Class     |
# |__________________________________________________________|


class ABSASentimentClassifier(pl.LightningModule):

  def __init__(self, config, **kwargs):
    super().__init__()
    self.config = config
    self.learning_rate = self.config['lr']
    self.save_hyperparameters('learning_rate', 'max_epochs')
    self.model = Get_Model(self.config['model'])

  
  def training_step(self, batch, batch_nb):

    # Forward
    y_hat = self.model(batch)

    # Loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))

    # Logs
    self.log_dict({'training_loss':loss}, prog_bar=True)

    return loss

  
  def validation_step(self, batch, batch_nb):
    
    # Forward
    y_hat = self.model(batch)
        
    # Loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))

    # Acc
    a, y_hat = torch.max(y_hat, dim=1)
    val_acc = accuracy_score(y_hat.cpu(), batch['label'].cpu())
    val_acc = torch.tensor(val_acc)
    
    # Logs
    self.log_dict({'val_loss':loss,'val_acc':val_acc}, prog_bar=True)
    
    return loss

  
  def test_step(self, batch, batch_nb):
    
    # Forward
    y_hat = self.model(batch)
      
    # Loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))
    
    a, y_hat = torch.max(y_hat, dim=1)
    test_acc = accuracy_score(y_hat.cpu(), batch['label'].cpu())
    
    # Logs
    self.log_dict({'test_loss':loss,'test_acc':test_acc}, prog_bar=True)

    return loss

  
  def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):

    return self.model(batch)


  '''
  |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
  | Training Setup  |
  |_________________|
  '''
  def configure_optimizers(self):
    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    |   REQUIRED                                                            |
    |   can return multiple optimizers and learning_rate schedulers         |
    |   (LBFGS it is automatically supported, no need for closure function) |
    |_______________________________________________________________________|
    '''
    optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
    scheduler = {   
      'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, 
                                                       steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                                                       epochs=self.hparams.max_epochs),
                 
      'interval': 'step'  # called after each training step
    } 

    #scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, cycle_momentum=False,step_size_up=300)
    #scheduler = ReduceLROnPlateau(optimizer, patience=0, factor=0.2)
    
    return [optimizer], [scheduler]

  @staticmethod
  def add_model_specific_args(self, parent_parser):  # pragma: no-cover
    """
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | Define parameters that only apply to this model     |
    |_____________________________________________________|
    """
    parser = ArgumentParser(parents=[parent_parser])

    # network params
    #parser.add_argument('--drop_prob', default=0.2, type=float)

    # data
    parser.add_argument('--data_root', default=os.path.join(os.getcwd(), 'train_val_data'), type=str)

    # training params (opt)
    parser.add_argument('--learning_rate', default= self.config['lr'], type=float, help = "type (default: %(default)f)")
    return parser