import math, os, argparse, json
from distutils.util import strtobool
from dataclasses import dataclass

import torch
import numpy as np

import pytorch_lightning as pl

from ablang_train import ablang_vocab
from .initial_models import AbLangPaired_v1


def ablang_parse_args(args=None, is_test=False):
    """
    Creates a parser. Then adds all pytorch-lightning arguments to it. Then adds model specific arguments to it.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', '-n', type=str, default="Model", help='Model name.')
    parser.add_argument('--json_args', type=str, default="", help='Model arguments in a json file.')
    parser = AbLangPaired_v1.add_model_specific_args(parser)
    parser = AbLangPaired_v1.add_training_specific_args(parser)
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    
    args = set_json_arguments(args, args.json_args)
    
    args.devices = int(args.devices) if args.devices else args.devices
    
    arguments = PrepareArguments(args, is_test)()
    
    return arguments

@dataclass
class ModelArguments:
    program_args:None
    trainer_args:None
    model_specific_args:None
    
    
def set_json_arguments(args, json_file):
    
    if json_file == "":
        return args
    
    with open(json_file, 'r') as f:
        new_arguments = json.load(f)
        
    for key, val in new_arguments.items():
        setattr(args, key, val)
    
    return args


class PrepareArguments:
    
    def __init__(self, args, is_test=False):
        
        if is_test:
            args.effective_batch_size = 16
            args.val_check_interval = 1
        else:
            args.val_check_interval=100
        
        self.args = args
    
    def set_vocab_args(self):
            
        self.args.pad_tkn = ablang_vocab['-']
        self.args.start_tkn = ablang_vocab['<']
        self.args.end_tkn = ablang_vocab['>']
        self.args.sep_tkn = ablang_vocab['|']
        self.args.mask_tkn = ablang_vocab['*']
        self.args.vocab_size = len(ablang_vocab)

    def _set_n_accummulated_grad_batches(self):
        
        gpu_batch_size = self.args.effective_batch_size // self.args.devices # Spread effective batch size across GPUs and gradient accumulation
        
        if int(gpu_batch_size // self.args.max_fit_batch_size) > 1: # Calculates how many times the gradients needs to be accumulated
            self.args.accumulate_grad_batches = int(gpu_batch_size // self.args.max_fit_batch_size)
            self.args.val_check_interval = int(self.args.val_check_interval * self.args.accumulate_grad_batches) # Adjust val check
            self.args.num_training_steps = int(self.args.num_training_steps * self.args.accumulate_grad_batches) # Adjust training steps            
        
    def set_device_arguments(self):
        """
        Hyparameters scale weirdly with gpus. This function is to adjust them based on gpu_counts.
        
        The following link provides a discussion for setting effective batch size and learning rate:
        https://forums.pytorchlightning.ai/t/effective-learning-rate-and-batch-size-with-lightning-in-ddp/101/8
        """ 
        
        if self.args.accelerator == 'cuda':
            self.args.precision = 16
            self._set_n_accummulated_grad_batches()
                
            # You LR*(gradient/gpus), and therefore you need to multiply your given LR with the number of gpus to get the effective LR
            self.args.learning_rate = self.args.learning_rate / self.args.devices
            if self.args.devices > 1:
                self.args.strategy = "ddp" #DDPPlugin() # find_unused_parameters=False 
            
        else:
            self.args.accumulate_grad_batches = 1
            
        self.args.train_batch_size = self.args.max_fit_batch_size  # We set the training batch size to be the max possible batch size
        self.args.max_steps = self.args.num_training_steps  # We set the training batch size to be the max possible steps        

    def set_trainer_args(self):
        """
        Used to update trainer arguments
        """
        hparamstmp = {**vars(self.args)}
        trainer_args = pl.Trainer.default_attributes()
        del trainer_args['callbacks'] # Do this for easier coding with callbacks

        for key, value in trainer_args.items():
            trainer_args[key] = hparamstmp[key]
        
        self.trainer_args = trainer_args
        
    def __call__(self):
        
        
        self.set_vocab_args()
        self.set_device_arguments()
        self.set_trainer_args()
        
        return ModelArguments(program_args='', trainer_args=self.trainer_args, model_specific_args=self.args)