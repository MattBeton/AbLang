import os
import numpy as np
import torch
import pytorch_lightning as pl

from loss_updating import loss_fn, optimizers, schedulers
from evaluation.evaluation import Evaluations


class TrainingFrame(pl.LightningModule):
    """
    Python-Lightning module for pretraining.
    
    This module controls training, with training_step automatically (and hidden) doing .zero_grad(), .backward() and .step() for optimizer and scheduler.
    
    """

    def __init__(self, conf, model, tokenizers):
        super().__init__()
        self.save_hyperparameters(conf) # saves to self.hparams
        
        self.loss_fn = loss_fn.get_loss_fn('CrossEntropy_Loss')()
        self.val_loss_fn = loss_fn.get_loss_fn('CrossEntropy_Loss')()
        self.tokenizer = tokenizers.ABtokenizer(os.path.join(self.hparams.data_path,'vocab.json'))
        
        self.AbLang = model.AbLang(self.hparams)
        self.Evaluations = Evaluations()

    def forward(self, x, attention_mask=None):
        
        output = self.AbLang(x, attention_mask)
        
        return output

    def training_step(self, dataset, batch_idx): 
        
        data, labels, attention_mask = dataset['input'], dataset['labels'], dataset['attention_mask']
        
        output = self(data, attention_mask=attention_mask)

        loss = self.loss_fn(output.view(-1, self.hparams.vocab_size), labels)
        
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
            
        # Only log once every global step
        if batch_idx % self.hparams.accumulate_grad_batches == 0:
            self.logger.experiment['evaluation/train_loss'].log(loss)
        
        return {"loss": loss}
    
    def training_step_end(self, batch_parts):
        
        return {'loss': batch_parts['loss'].mean()}
    
    def validation_step(self, dataset, batch_idx): # Updated every step when validation is called
        
        data, labels = dataset['input'], dataset['labels']
        
        output = self(data, attention_mask=None)
        
        loss = self.val_loss_fn(output.view(-1, self.hparams.vocab_size), labels)
        
        all_loss = self.val_loss_fn(output.view(-1, self.hparams.vocab_size), labels, reduce=False).view(output.shape[:-1])[:, 1:]
                
        fw1_loss = all_loss[:, :27+1].mean()
        cdr1_loss = all_loss[:, 27:38+1].mean()
        fw2_loss = all_loss[:, 38:56+1].mean()
        cdr2_loss = all_loss[:, 56:65+1].mean()
        fw3_loss = all_loss[:, 65:105+1].mean()
        cdr3_loss = all_loss[:, 105:117+1].mean()
        fw4_loss = all_loss[:, 117:].sum() / (len(all_loss[:, 117:].reshape(-1)) - (labels == -100).sum())

        return {'val_loss': loss, 'fw1_loss':fw1_loss, 'cdr1_loss':cdr1_loss, 'fw2_loss':fw2_loss, 
                'cdr2_loss':cdr2_loss, 'fw3_loss':fw3_loss, 'cdr3_loss':cdr3_loss, 'fw4_loss':fw4_loss}
    
    
    def validation_epoch_end(self, val_step_outputs): # Updated once when validation is called
        
        self.Evaluations(self, val_step_outputs)

    
    def configure_optimizers(self):
        """
        Initialization of the optimizer and scheduler used for training
        """
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                                         "weight_decay": self.hparams.weight_decay,},
                                        {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                                         "weight_decay": 0.0,},]
        
        optimizer = optimizers.AdamW(optimizer_grouped_parameters, 
                                     lr=self.hparams.learning_rate, 
                                     betas = self.hparams.adam_betas, 
                                     eps=self.hparams.adam_epsilon, 
                                     weight_decay=self.hparams.weight_decay, 
                                     correct_bias = True)
        
        scheduler = schedulers.get_cosine_schedule_with_warmup(optimizer, 
                                                               num_warmup_steps=self.hparams.warmup_steps,
                                                               num_training_steps=self.hparams.num_training_steps)
   
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval':'step',
                #"monitor": "train_loss",
                "frequency":1,
            },
        }



