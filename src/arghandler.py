import math, os, argparse, json
from distutils.util import strtobool

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.neptune import NeptuneLogger


def parse_args(args=None, test=False):
    """
    Creates a parser. Then adds all pytorch-lightning arguments to it. Then adds model specific arguments to it.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', '-n', type=str, default="Model", help='Name of your model')
    parser.add_argument('--dataDir', '-d', type=str, default='../data/11022022_data', help='Path to data containing directory')
    parser.add_argument('--outpath', type=str, default="/data/iraqbabbler/olsen/Documents/projects/AbLang/train_ablang/reports")
    
    parser.add_argument('--cpus', type=int, default=1, help='Number of cpus to use on data handling (4xGPUs is the recommended). \
                                                                0 uses the main process to load the data.')
    
    parser.add_argument('--max_batch_size', type=int, default=256, help='Max batch size that fits in GPU memory')
    parser.add_argument('--effective_batch_size', type=int, default=4_096*2, help='Effective batch size')
    parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-04, help='Learning rate')
    parser.add_argument('--cdr3_focus', type=float, default=1, help='Used to increase the chance of masking the CDR3 region. \
                                                                        1 is same as other residues, \
                                                                        2 is 2 times the chance, \
                                                                        3 is 3 times the chance, etc..')
    
    parser.add_argument('--chain', type=str, default='heavy', help='Chain you are training on')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden layer size')
    parser.add_argument('--adam_epsilon', type=float, default=1e-7, help='Adam Epsilon')
    parser.add_argument('--mask_percent', type=float, default=0.25, help='Maskinig percentage')
    parser.add_argument('--mask_variable', type=strtobool, default=True, help='Masking be uniform random between 0 and mask_percent for each batch')
    
    
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args(args)
    
    if test:
        args.effective_batch_size = 16
        args.val_check_interval = 1
    else:
        args.val_check_interval=100
        
    args.seed = 42 # Set seed
    
    model_arguments = PrepareArguments(args)
    
    model_arguments.add_model_hparams()
    model_arguments.add_training_args()
    model_arguments.set_neptune_logger()
    model_arguments.set_trainer_args()
    
    #hparams.sync_batchnorm=True # Slower training, but might improve training
    
    return model_arguments.trainer_args, model_arguments.hparams


class PrepareArguments:
    
    def __init__(self, args):
        
        self.args = {**vars(args)}
    
    def add_model_hparams(self):
        """
        Sets the hparams from the default values in a given json file 
        and overwrites these with the values given from argparser.
        """
        
        with open(os.path.join(self.args['dataDir'], 'default_hparams.json')) as hparams_file: 
            hparams = json.load(hparams_file)

        for key, value in self.args.items(): 
            hparams[key] = value

        hparams['intermediate_size'] = 4 * hparams['hidden_size']

        with open(os.path.join(self.args['dataDir'], 'vocab.json')) as vocab_file: 
            vocab = json.load(vocab_file)
            hparams['pad_token_id'] = vocab['-']
            hparams['start_token_id'] = vocab['<']
            hparams['stop_token_id'] = vocab['>']
            hparams['mask_token_id'] = vocab['*']
            hparams['divide_token_id'] = vocab['|']
            hparams['vocab_size'] = len(vocab)

        self.hparams = argparse.Namespace(**hparams)

    def add_training_args(self):
        """
        Add training args to hparam
        """ 
        
        hparams = self.hparams

        ###### hparams scale weirdly with gpus #######
        try:
            if hparams.gpus > 1:
                gpu_count = hparams.gpus
                hparams.strategy = DDPPlugin(find_unused_parameters=False)
                hparams.precision = 16
            elif hparams.gpus == 1:
                gpu_count = hparams.gpus
                hparams.precision = 16
            else:
                gpu_count = 1
        except:
            gpu_count = 1

        ###### Spread effective batch size across GPUs and gradient accumulation #######
        gpu_batch_size = hparams.effective_batch_size // gpu_count
        hparams.accumulate_grad_batches = int(gpu_batch_size // hparams.max_batch_size) # Calculates how many times the gradients needs to be accumulated
        if hparams.accumulate_grad_batches < 1:
            hparams.accumulate_grad_batches = 1

        hparams.val_check_interval = int(hparams.val_check_interval * hparams.accumulate_grad_batches)

        hparams.train_batch_size = hparams.max_batch_size  # The training batch size is the max possible batch size
        hparams.max_steps = hparams.num_training_steps  # The training batch size is the max possible batch size

        # You LR*(gradient/gpus), and therefore you need to divide your given LR with the number of gpus to get the effective LR
        hparams.lr = hparams.lr / gpu_count 

        hparams.warmup_steps = int(hparams.max_steps * 0.05)

        self.hparams = hparams
        
    def set_neptune_logger(self):
        """
        Initialize Neptune logger
        """

        neptune_args = { 'api_key':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZmVhYTY2NzAtOGUxYS00NWFlLWI0MDQtZjM5ODBmYmNkMjA3In0=",
        'project':"tobiasheol/AbLangTraining",
        'name':self.hparams.name,
        'log_model_checkpoints':False,
        }
        
        self.logger = NeptuneLogger(**neptune_args)

    def set_trainer_args(self):
        """
        Used to update trainer arguments
        """
        hparamstmp = {**vars(self.hparams)}
        trainer_args = pl.Trainer.default_attributes()
        del trainer_args['callbacks'] # Do this for easier coding with callbacks
        del trainer_args['progress_bar_refresh_rate']

        for key, value in trainer_args.items():
            trainer_args[key] = hparamstmp[key]
        
        trainer_args['logger'] = self.logger
        
        self.trainer_args = trainer_args