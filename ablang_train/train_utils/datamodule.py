import os
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .datacollators import ABcollator


class AbDataModule(pl.LightningDataModule):

    def __init__(self, data_hparams, tokenizer):
        super().__init__()
        self.data_hparams = data_hparams
        self.tokenizer = tokenizer()        
        
    def setup(self, stage=None): # called on every GPU
        
        self.traincollater = ABcollator(
            self.tokenizer, 
            pad_tkn = self.data_hparams.pad_tkn,
            start_tkn = self.data_hparams.start_tkn,
            end_tkn = self.data_hparams.end_tkn,
            sep_tkn = self.data_hparams.sep_tkn,
            mask_tkn = self.data_hparams.mask_tkn,
            mask_percent=self.data_hparams.mask_percent,
            mask_variable=self.data_hparams.variable_masking,
            cdr3_focus=self.data_hparams.cdr3_focus,
            mask_technique=self.data_hparams.mask_technique,
        )
        
        self.evalcollater = ABcollator(
            self.tokenizer, 
            pad_tkn = self.data_hparams.pad_tkn,
            start_tkn = self.data_hparams.start_tkn,
            end_tkn = self.data_hparams.end_tkn,
            sep_tkn = self.data_hparams.sep_tkn,
            mask_tkn = self.data_hparams.mask_tkn,
            mask_percent=self.data_hparams.mask_percent,
            mask_variable=self.data_hparams.variable_masking,
            cdr3_focus=1.,
            mask_technique=self.data_hparams.mask_technique,
        )
        
        self.train = self.get_data(
            file_path=os.path.join(self.data_hparams.data_path,'train_data'),
            over_sample_data=self.data_hparams.over_sample_data
        )
        self.val = self.get_data(file_path=os.path.join(self.data_hparams.data_path,'eval_data'))
        
    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.data_hparams.train_batch_size, 
                          collate_fn=self.traincollater, 
                          num_workers=self.data_hparams.cpus,
                          shuffle=True,
                          pin_memory=True,
                         )

    def val_dataloader(self): # rule of thumb is: num_worker = 4 * num_GPU
        return DataLoader(self.val, 
                          batch_size=self.data_hparams.eval_batch_size, 
                          collate_fn=self.evalcollater, 
                          num_workers=self.data_hparams.cpus,
                          pin_memory=True,
                         )


    def get_data(self, file_path, over_sample_data=False):
        "Reads txt file of sequences."
        
        if os.path.isfile(os.path.join(file_path,'heavy_chains.txt')):
            with open(os.path.join(file_path,'heavy_chains.txt'), encoding="utf-8") as f:
                heavychain = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        else:
            heavychain = []
            
        if os.path.isfile(os.path.join(file_path,'light_chains.txt')):
            with open(os.path.join(file_path,'light_chains.txt'), encoding="utf-8") as f:
                lightchain = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        else:
            lightchain = []
            
        if os.path.isfile(os.path.join(file_path,'paired_chains.txt')):
            with open(os.path.join(file_path,'paired_chains.txt'), encoding="utf-8") as f:
                pairedchain = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        else:
            pairedchain = []
            
        if over_sample_data:
            sizes = [len(heavychain), len(lightchain), len(pairedchain)]
            scale = (np.max(sizes)/sizes).astype(np.int16)
            
            return heavychain*scale[0] + lightchain*scale[1] + pairedchain*scale[2]
            
        else:
            return heavychain + lightchain + pairedchain