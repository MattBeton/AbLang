import math, os, argparse, json
from distutils.util import strtobool
from dataclasses import dataclass

import torch
import numpy as np

import pytorch_lightning as pl

from ablang_train import ablang_vocab
from .initial_models import AbLangPaired_v1


def ablang_parse_args(args=None):
    """
    Creates a parser. Then adds all pytorch-lightning arguments to it. Then adds model specific arguments to it.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', '-n', type=str, default="Model", help='Model name.')
    parser.add_argument('--json_args', type=str, default="", help='Model arguments in a json file.')
    parser = AbLangPaired_v1.add_model_specific_args(parser)
    parser = AbLangPaired_v1.add_training_specific_args(parser)
    parser = AbLangPaired_v1.add_pl_train_args(parser)
    
    #parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    
    args = set_json_arguments(args, args.json_args)
    
    args.devices = int(args.devices) if args.devices else args.devices
    
    arguments = PrepareArguments(args)()
    
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
    
    def __init__(self, args):
        
        self.args = args
    
    def set_vocab_args(self):
            
        self.args.pad_tkn = ablang_vocab['-']
        self.args.start_tkn = ablang_vocab['<']
        self.args.end_tkn = ablang_vocab['>']
        self.args.sep_tkn = ablang_vocab['|']
        self.args.mask_tkn = ablang_vocab['*']
        self.args.vocab_size = len(ablang_vocab)

    def _set_n_accummulated_grad_batches(self):
        
        # Spread effective batch size across GPUs and gradient accumulation
        gpu_batch_size = self.args.effective_batch_size // self.args.devices 
        accumulate_size = int(gpu_batch_size // self.args.max_fit_batch_size)
        
        if accumulate_size > 1: 
            # Calculates how many times the gradients needs to be accumulated
            self.args.accumulate_grad_batches = accumulate_size
            
            # Adjust val_check, n_log and training_steps
            self.args.val_check_interval = int(self.args.val_check_interval * accumulate_size) 
            self.args.log_every_n_steps = int(self.args.log_every_n_steps * accumulate_size) 
            #self.args.num_training_steps = int(self.args.num_training_steps * accumulate_size)
        
    def set_device_arguments(self):
        """
        Hyparameters scale weirdly with gpus. This function is to adjust them based on gpu_counts.
        
        The following link provides a discussion for setting effective batch size and learning rate:
        https://forums.pytorchlightning.ai/t/effective-learning-rate-and-batch-size-with-lightning-in-ddp/101/8
        """ 
        self._set_n_accummulated_grad_batches()
        
        if self.args.accelerator == 'cuda':
            self.args.precision = "16-mixed"  
            # You LR*(gradient/gpus), 
            # and therefore need to multiply your given LR with the number of gpus to get the effective LR
            self.args.learning_rate = self.args.learning_rate / self.args.devices
            if self.args.devices > 1:
                self.args.strategy = "ddp" #DDPPlugin() # find_unused_parameters=False 
        
        # We set the training batch size to be the max possible batch size
        self.args.train_batch_size = self.args.max_fit_batch_size  
        # We set the number of training steps to be the max possible steps
        self.args.max_steps = self.args.num_training_steps  

    def set_trainer_args(self):
        """
        Used to update trainer arguments
        """
        hparamstmp = {**vars(self.args)}
        
        trainer_args = {}
        trainer_keys = ['accelerator', 'devices', 'precision', 'strategy',
                        #'logger',
                        'val_check_interval', '', 'enable_checkpointing', 
                        'default_root_dir', 'max_steps', 'accumulate_grad_batches', 
                       ]

        for key in trainer_keys:
            if key in hparamstmp: trainer_args[key] = hparamstmp[key]
        
        self.trainer_args = trainer_args
        
        
    def set_neptune_logger(self, name):
        """
        Initialize Neptune logger
        """

        neptune_args = { 'api_key':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0N2Y2YmIxMS02OWM3LTRhY2MtYTQxOC0xODU5N2E0ODFmMzEifQ==",
        'project':"tobiasheol/AbLangTraining",
        'name':name,
        'log_model_checkpoints':False,
        }

        return pl.loggers.neptune.NeptuneLogger(**neptune_args)
        
    def __call__(self):
        
        
        self.set_vocab_args()
        self.set_device_arguments()
        self.set_trainer_args()
        self.trainer_args['logger'] = self.set_neptune_logger(self.args.name)
        
        return ModelArguments(program_args='', trainer_args=self.trainer_args, model_specific_args=self.args)