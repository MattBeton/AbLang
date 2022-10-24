from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from .n_step_checkpoint import CheckpointEveryNSteps


class CallbackHandler:
    
    def __init__(self, 
                 n_steps=100, 
                 progress_refresh_rate=0, 
                 outpath="/data/iraqbabbler/olsen/Documents/projects/AbLang/train_ablang_paired/reports"):
        
        self.callbacks = []
        
        self.callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        self.callbacks.append(ModelCheckpoint(save_last=False, save_top_k=0, dirpath=outpath))  #, dirpath=outpath
        
        self.callbacks.append(CheckpointEveryNSteps(n_steps))
 
        self.callbacks.append(TQDMProgressBar(refresh_rate=progress_refresh_rate))
        
    def __call__(self):
        
        return self.callbacks