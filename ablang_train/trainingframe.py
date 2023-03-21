import os
import numpy as np
import torch
import pytorch_lightning as pl
import math

from ablang_train import Evaluations
from ablang_train.train_utils.schedulers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from ablang_train.train_utils.loss_fn import get_loss_fn

class TrainingFrame(pl.LightningModule):
    """
    Python-Lightning module for pretraining.
    
    This module controls training, with training_step automatically (and hidden) doing .zero_grad(), .backward() and .step() for optimizer and scheduler.
    
    """

    def __init__(self, conf, model, tokenizer):
        super().__init__()
        self.save_hyperparameters(conf) # saves to self.hparams
        
        self.loss_fn = get_loss_fn(self.hparams.loss_fn)(gamma = self.hparams.fl_gamma) #torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer()
        
        self.ablang = model(
            vocab_size = self.hparams.vocab_size,
            hidden_embed_size = self.hparams.hidden_embed_size,
            n_attn_heads = self.hparams.n_attn_heads,
            n_encoder_blocks = self.hparams.n_encoder_blocks,
            padding_tkn = self.hparams.pad_tkn,
            mask_tkn = self.hparams.mask_tkn,
            layer_norm_eps = self.hparams.layer_norm_eps,
            dropout = self.hparams.dropout, 
            use_tkn_dropout = self.hparams.use_tkn_dropout,
        )
        self.ablang.apply(self._init_weights) # Initialize weights
        self.run_evaluations = Evaluations(self.tokenizer, self.hparams) # Initialize evaluations      
        
    def _init_weights(self, module):
        
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, tokens):
        
        return self.ablang(tokens)

    def training_step(self, dataset, batch_idx): 
        
        tokens, labels = dataset['input'], dataset['labels']
        
        logits = self(tokens)
        loss = self.loss_fn(logits.view(-1, self.hparams.vocab_size), labels)
        
        # Must clear cache at regular interval
        if (batch_idx % self.hparams.accumulate_grad_batches) % 10 == 0: #self.global_step 
            torch.cuda.empty_cache()
        
        return {"loss": loss}
    
    def training_step_end(self, batch_parts):
        
        train_loss = batch_parts['loss'].mean()
        self.logger.experiment['evaluation/train_loss'].log(train_loss)
        
        return {'loss': train_loss}
    
    def validation_step(self, dataset, batch_idx): # Updated every step when validation is called
        
        tokens, labels = dataset['input'], dataset['labels']
        loss, perplexity = self.run_evaluations.loss_n_perplexity.calculate_perplexity_fast(self, dataset['sequences'])
        
        heavy, light = [], []
        for record in dataset['sequences']:
            h, l = record.split('|')
            
            heavy.append(h)
            light.append(l)
        
        loss_heavy, perplexity_heavy = self.run_evaluations.loss_n_perplexity.calculate_perplexity_fast(self, heavy)
        loss_light, perplexity_light = self.run_evaluations.loss_n_perplexity.calculate_perplexity_fast(self, light)

        return {
            'val_loss': loss, 'perplexity':perplexity, 
            'val_loss_h': loss_heavy, 'perplexity_h':perplexity_heavy,
            'val_loss_l': loss_light, 'perplexity_l':perplexity_light,
        }
    
    
    def validation_epoch_end(self, val_step_outputs): # Updated once when validation is called
        
        perplexity = torch.stack([x['perplexity'] for x in val_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity"].log(perplexity)
        
        val_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss"].log(val_loss)
        
        perplexity = torch.stack([x['perplexity_h'] for x in val_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity_h"].log(perplexity)
        
        val_loss = torch.stack([x['val_loss_h'] for x in val_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss_h"].log(val_loss)
        
        perplexity = torch.stack([x['perplexity_l'] for x in val_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity_l"].log(perplexity)
        
        val_loss = torch.stack([x['val_loss_l'] for x in val_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss_l"].log(val_loss)
        
        self.run_evaluations(self)

    
    def configure_optimizers(self):
        """
        Initialization of the optimizer and scheduler used for training
        """
        model = self.ablang
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                     lr=self.hparams.learning_rate, 
                                     betas = self.hparams.adam_betas, 
                                     eps=self.hparams.adam_epsilon, 
                                     weight_decay=self.hparams.weight_decay, 
                                     )
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.hparams.num_training_steps
                                                   )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

