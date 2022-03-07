import math, os, argparse, json
from distutils.util import strtobool
from dataclasses import dataclass

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.neptune import NeptuneLogger

from .initial_models import AbLangPaired_v1


def parse_args(args=None, is_test=False):
    """
    Creates a parser. Then adds all pytorch-lightning arguments to it. Then adds model specific arguments to it.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', '-n', type=str, default="Model", help='Model name.')
    parser.add_argument('--cpus', type=int, default=1, help='Number of cpus to use on data handling (4xGPUs is the recommended). \
                                                                    0 uses the main process to load the data.')
    
    parser = AbLangPaired_v1.add_model_specific_args(parser)
    parser = AbLangPaired_v1.add_training_specific_args(parser)
    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    
    arguments = PrepareArguments(args, is_test)()
    
    return arguments

@dataclass
class ModelArguments:
    program_args:None
    trainer_args:None
    model_specific_args:None
    

class PrepareArguments:
    
    def __init__(self, args, is_test=False):
        
        if is_test:
            args.effective_batch_size = 16
            args.val_check_interval = 1
        else:
            args.val_check_interval=100
        
        self.args = args
    
    def set_vocab_args(self):

        with open(os.path.join(self.args.data_path, 'vocab.json')) as vocab_file: 
            vocab = json.load(vocab_file)
            
        self.args.pad_token_id = vocab['-']
        self.args.start_token_id = vocab['<']
        self.args.stop_token_id = vocab['>']
        self.args.mask_token_id = vocab['*']
        self.args.split_token_id = vocab['|']
        self.args.vocab_size = len(vocab)

    def set_gpus(self):
        """
        Add training args to hparam
        """ 
        try:
            if self.args.gpus > 1:
                self.args.gpu_count = self.args.gpus
                self.args.strategy = DDPPlugin(find_unused_parameters=False)
                self.args.precision = 16
            elif self.args.gpus == 1:
                self.args.gpu_count = self.args.gpus
                self.args.precision = 16
            else:
                self.args.gpu_count = 1
        except:
            self.args.gpu_count = 1

    def set_gpus_settings(self):
        """
        Hyparameters scale weirdly with gpus. This function is to adjust them based on gpu_counts.
        
        The following link provides a discussion for setting effective batch size and learning rate:
        https://forums.pytorchlightning.ai/t/effective-learning-rate-and-batch-size-with-lightning-in-ddp/101/8
        """

        gpu_batch_size = self.args.effective_batch_size // self.args.gpu_count # Spread effective batch size across GPUs and gradient accumulation
        
        if int(gpu_batch_size // self.args.max_fit_batch_size) > 1: # Calculates how many times the gradients needs to be accumulated
            self.args.accumulate_grad_batches = int(gpu_batch_size // self.args.max_fit_batch_size)
            self.args.val_check_interval = int(self.args.val_check_interval * self.args.accumulate_grad_batches) # Adjust val check
        else:
            self.args.accumulate_grad_batches = 1

        self.args.train_batch_size = self.args.max_fit_batch_size  # We set the training batch size to be the max possible batch size
        self.args.max_steps = self.args.num_training_steps  # We set the training batch size to be the max possible steps

        # You LR*(gradient/gpus), and therefore you need to multiply your given LR with the number of gpus to get the effective LR
        self.args.learning_rate = self.args.learning_rate #/ self.args.gpu_count 

        self.args.warmup_steps = int(self.args.max_steps * 0.05)
        
    def set_neptune_logger(self):
        """
        Initialize Neptune logger
        """

        neptune_args = { 'api_key':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZmVhYTY2NzAtOGUxYS00NWFlLWI0MDQtZjM5ODBmYmNkMjA3In0=",
        'project':"tobiasheol/AbLangTraining",
        'name':self.args.name,
        'log_model_checkpoints':False,
        }
        
        self.args.logger = NeptuneLogger(**neptune_args)

    def set_trainer_args(self):
        """
        Used to update trainer arguments
        """
        hparamstmp = {**vars(self.args)}
        trainer_args = pl.Trainer.default_attributes()
        del trainer_args['callbacks'] # Do this for easier coding with callbacks
        del trainer_args['progress_bar_refresh_rate']

        for key, value in trainer_args.items():
            trainer_args[key] = hparamstmp[key]
        
        trainer_args['logger'] = self.args.logger
        
        self.trainer_args = trainer_args
        
    def __call__(self):
        
        
        self.set_vocab_args()
        self.set_gpus()
        self.set_gpus_settings()
        self.set_neptune_logger()
        self.set_trainer_args()
        
        return ModelArguments(program_args='', trainer_args=self.trainer_args, model_specific_args=self.args)