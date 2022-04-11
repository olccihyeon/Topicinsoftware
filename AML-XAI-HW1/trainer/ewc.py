from __future__ import print_function

import copy
import logging

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
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)
        
        self.lamb=args.lamb
    
    

    def train(self, train_loader, test_loader, t, device = None):
        
        self.device = device
   
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0: # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()
        
        # Now, you can update self.t
        self.t = t
        
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        
        
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        
    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """
        
        #######################################################################################
        
        loss =0
        celoss = nn.CrossEntropyLoss()
        loss += celoss(output,targets)
        # Write youre code here
        fisher_loss = 0
        if self.t>0:
            model_fixed = {n: p.clone().detach() for n, p in self.model_fixed.named_parameters()
                    }
            for n, p in self.model.named_parameters():
                _loss = torch.sum(self.fisher[n] * ((p - model_fixed[n]) ** 2 ))
                fisher_loss += _loss
            loss += (self.lamb/2) * fisher_loss
        return loss
        
        
        #######################################################################################
    
    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix. 
        
        This function will be used in the function 'update_fisher'
        """
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        
        #######################################################################################
   
        celoss = nn.CrossEntropyLoss()
        for samples in self.fisher_iterator:
            data, target = samples
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]
            self.model.zero_grad()
            output = self.model(data)[self.t]
            loss = celoss(output,target)
       
            (loss).backward()

            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=(batch_size)**2*(p.grad**2)
            #20은 batch size of fisher
        

           
        fisher = {n: p/len(self.fisher_iterator.dataset) for n, p in fisher.items()}
        return fisher

        

        # Write youre code here
        
        
        
        #######################################################################################        
    
    def update_fisher(self):
        
        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """
        
        #######################################################################################
        if self.t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.compute_diag_fisher()
        if self.t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*(self.t))/(self.t+1)
        
        
        
        
        #######################################################################################


