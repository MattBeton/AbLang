import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .datacollators import ABcollator


class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_hparams, tokenizer):
        super().__init__()
        self.data_hparams = data_hparams
        self.tokenizer = tokenizer.ABtokenizer(os.path.join(data_hparams.data_path,'vocab.json'))
        
    def setup(self, stage=None): # called on every GPU
        
        self.traincollater = ABcollator(self.tokenizer, 
                                        pad_to_mask=self.data_hparams.pad_token_id, 
                                        mask_percent=self.data_hparams.mask_percent,
                                        mask_variable=self.data_hparams.variable_masking,
                                        cdr3_focus=self.data_hparams.cdr3_focus
                                       )
        
        self.evalcollater = ABcollator(self.tokenizer, 
                                       pad_to_mask=self.data_hparams.pad_token_id, 
                                       mask_percent=0,
                                       mask_variable=False,
                                       cdr3_focus=1
                                      )
        
        
        self.train = self.get_data(file_path=os.path.join(self.data_hparams.data_path,'train_data'))
        self.val = self.get_data(file_path=os.path.join(self.data_hparams.data_path,'eval_data'))[:1000]

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


    def get_data(self, file_path):
        "Reads txt file of sequences."
        
        with open(os.path.join(file_path,'heavy_chains.txt'), encoding="utf-8") as f:
            heavychain = [line + '|' for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        with open(os.path.join(file_path,'light_chains.txt'), encoding="utf-8") as f:
            lightchain = ['|' + line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        with open(os.path.join(file_path,'paired_chains.txt'), encoding="utf-8") as f:
            pairedchain = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            
        return heavychain + lightchain + pairedchain