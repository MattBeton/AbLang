import os, json
from glob import glob
import time
import datetime

import torch
import pytorch_lightning as pl

from torch.nn.parallel import DistributedDataParallel

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    
    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, *args):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        step = trainer.global_step        
        
        if (batch_idx / trainer.accumulate_grad_batches) % self.save_step_frequency == 0:

            if not trainer.logger.version:
                correct_dirpath = os.path.join(trainer.checkpoint_callback.dirpath, 'tmp_model')
            else:
                correct_dirpath = os.path.join(trainer.checkpoint_callback.dirpath, trainer.logger.version)
            
            os.makedirs(correct_dirpath, exist_ok=True)
            
            
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
                
            else:             
                filename = "epoch={}-step={}.ckpt".format(epoch, step)
                
            
            for ckpt_file in glob(os.path.join(correct_dirpath, '*')):
                try:
                    os.remove(ckpt_file)
                except OSError:
                    pass
            
            amodel = pl_module.AbLang
            tokenizer = pl_module.tokenizer
            
            #### SAVE THE MODEL CHECKPOINT ##### (saving both the checkpoint and the model state dict might be a little redundant)
            ckpt_path = os.path.join(correct_dirpath, filename)

            #trainer.save_checkpoint(ckpt_path)
            
            #### SAVE THE STATE DICT OF THE MODEL #####
            torch.save(amodel.state_dict(), os.path.join(correct_dirpath, 'amodel.pt'))
            
            #### SAVE THE USED HPARAMS #####
            with open(os.path.join(correct_dirpath, 'hparams.json'), 'w', encoding='utf-8') as f:
                
                hparamstmp = amodel.hparams
                hparamstmp.logger = 'neptune'
                hparamstmp.strategy = 'not logged'
                hparamstmp.tpu_cores = 'not logged'
                
                json.dump(dict(hparamstmp), f, ensure_ascii=False)
                
            #### SAVE THE USED VOCAB FILE #####
            with open(os.path.join(correct_dirpath, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(tokenizer.vocab_to_token, f, ensure_ascii=False)


                
"""
class TrainingTimeCallback(pl.Callback):
    #
    Class that saves how long it takes to train a batch
    #
    
    def __init__(self):
        self.start_time = time.perf_counter()
    
    def on_validation_start(self, trainer: pl.Trainer, pl_module):
        
        if (trainer.batch_idx / trainer.accumulate_grad_batches) % 1 == 0:
            
            print(trainer.batch_idx)
            
            batch_seconds = time.perf_counter() - self.start_time
            sample1000_seconds = (batch_seconds/trainer.get_model().hparams.train_batch_size)*1000
        
            batch_time = datetime.timedelta(seconds=batch_seconds)
            sample1000_time = datetime.timedelta(seconds=sample1000_seconds)
            
            trainer.logger.experiment.log_metric('batch_idx', trainer.batch_idx)
            trainer.logger.experiment.log_text('batch_time', 'A single batch: {} - 1000 sequences: {}'.format(str(batch_time), str(sample1000_time)))
            self.start_time = time.perf_counter()
"""