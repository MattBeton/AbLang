import os, json, glob

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class CallbackHandler:
    
    def __init__(self, 
                 save_step_frequency=100, 
                 progress_refresh_rate=0, 
                 outpath="/data/iraqbabbler/olsen/Documents/projects/AbLang/train_ablang_paired/reports"):
        
        self.callbacks = []
        
        self.callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        self.callbacks.append(ModelCheckpoint(save_last=False, save_top_k=0, dirpath=outpath))
        
        self.callbacks.append(CheckpointEveryNSteps(save_step_frequency))
 
        self.callbacks.append(TQDMProgressBar(refresh_rate=progress_refresh_rate))
        
    def __call__(self):
        
        return self.callbacks
    

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
                
            
            for ckpt_file in glob.glob(os.path.join(correct_dirpath, '*')):
                try:
                    os.remove(ckpt_file)
                except OSError:
                    pass
            
            model = pl_module.ablang
            tokenizer = pl_module.tokenizer
            
            #### SAVE THE MODEL CHECKPOINT ##### (saving both the checkpoint and the model state dict might be a little redundant)
            ckpt_path = os.path.join(correct_dirpath, filename)

            #trainer.save_checkpoint(ckpt_path)
            
            #### SAVE THE STATE DICT OF THE MODEL #####
            torch.save(model.state_dict(), os.path.join(correct_dirpath, 'model.pt'))
            
            #### SAVE THE USED HPARAMS #####
            with open(os.path.join(correct_dirpath, 'hparams.json'), 'w', encoding='utf-8') as f:
                
                hparamstmp = pl_module.hparams
                hparamstmp.logger = 'neptune'
                hparamstmp.strategy = 'not logged'
                hparamstmp.tpu_cores = 'not logged'
                
                json.dump(dict(hparamstmp), f, ensure_ascii=False)
                
            #### SAVE THE USED VOCAB FILE #####
            with open(os.path.join(correct_dirpath, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(tokenizer.vocab_to_token, f, ensure_ascii=False)