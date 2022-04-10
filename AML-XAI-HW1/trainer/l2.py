from __future__ import print_function

import copy
import logging
from tkinter.tix import CELL

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)
        
        self.lamb=args.lamb
        

    def train(self, train_loader, test_loader, t, device = None):
        
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        # if t>0: # update fisher before starting training new task
        #     self.update_frozen_model()
        
        
        # Now, you can update self.t
        self.t = t
        
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        task_params = {}
        if t >0:
            for n, p in self.model.named_parameters():
                if n[0:4] != 'last':
                    task_params[n] = p.clone().detach()
        
        
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target,task_params)
           

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        
    def criterion(self,output,targets,task_params):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """
        
        #######################################################################################
        
        # Write youre code here
        if not task_params:
          
            celoss = nn.CrossEntropyLoss()
            loss = celoss(output,targets)
            return loss
        else :
            reg_task_loss=0
            for n, p in self.model.named_parameters():
                if n[0:4] != 'last':
                
                    reg_task_loss += ((p - task_params[n])**2).sum()
            #self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
            loss_ce = nn.CrossEntropyLoss()
            CEloss = loss_ce(output,targets)
            loss = CEloss + self.lamb*reg_task_loss
            return loss
            


        
        
        #######################################################################################
        
