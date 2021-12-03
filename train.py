import argparse
import json
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

from models.model import ABSAModel_Bert
from src.dataloader import ABSADataModule
from src.utils import get_training_args

with open("setup.json") as setupfile:
    setup = json.load(setupfile)

# |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# | Define the Pytorch Lightning Module Classifier Class     |
# |__________________________________________________________|

class ABSASentimentClassifier(pl.LightningModule):

  def __init__(self, learning_rate = setup['lr'], weights=None, **kwargs):
    super().__init__()

    self.save_hyperparameters('learning_rate', 'max_epochs')
    self.model = ABSAModel_Bert()
    self.weights = weights
    self.preds = []

  def forward(self, x):
    y_hat = self.model(x)

    return y_hat
  
  def training_step(self, batch, batch_nb):

    # Forward
    y_hat = self.model(batch)

    # if self.weights:
    #   self.weights = torch.tensor(class_weights,dtype=torch.float) 
    
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
    self.model.eval()
    
    # Forward
    yhat = self.model(batch)
    self.preds = self.preds.extend(yhat.cpu().detach().numpy().tolist())
    return 

  
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
    
    return [optimizer], [scheduler]

  @staticmethod
  def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
    """
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | Define parameters that only apply to this model     |
    |_____________________________________________________|
    """
    parser = ArgumentParser(parents=[parent_parser])

    # data
    parser.add_argument('--data_root', default=os.path.join(root_dir, 'train_val_data'), type=str)

    # training params (opt)
    parser.add_argument('--learning_rate', default=setup['lr'], type=float, help = "type (default: %(default)f)")

    return parser



def cli_main():
    
    _, args, early_stop, checkpoint_callback, lr_monitor = get_training_args(setup) 

    ''' 
    Main training routine specific for this project. 
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | 1 INIT LIGHTNING MODEL    |
    |___________________________|
    '''

    print(args)

    if (vars(args)['model_load']):
      model = ABSASentimentClassifier.load_from_checkpoint(vars(args)['model_name'])
    else:  
      model = ABSASentimentClassifier(**vars(args))


    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | 3 INIT Trainer   |
    |__________________|
    '''
    trainer = Trainer.from_argparse_args(args,
        callbacks=[early_stop,lr_monitor, checkpoint_callback]
        )    

    absa_dm = ABSADataModule()

    '''
    |¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
    | 4 START Training   |
    |____________________|
    '''
    trainer.fit(model,absa_dm)


if __name__ == '__main__':
    cli_main()


