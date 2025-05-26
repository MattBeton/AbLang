import os
import numpy as np
import torch
import pytorch_lightning as pl
import math

from ablang_train import Evaluations
from ablang_train.train_utils.schedulers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from ablang_train.train_utils.loss_fn import get_loss_fn
from ablang_train.kfac_utils import KFACLinear
from ablang_train.koap import KOAP

def replace_linear_with_kfaclinear(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, torch.nn.Linear):
            kfac_linear_layer = KFACLinear(child_module.in_features, child_module.out_features, bias=child_module.bias is not None)
            kfac_linear_layer.weight = child_module.weight
            if child_module.bias is not None:
                kfac_linear_layer.bias = child_module.bias
            setattr(module, name, kfac_linear_layer)
        else:
            replace_linear_with_kfaclinear(child_module)

class TrainingFrame(pl.LightningModule):
    """
    Python-Lightning module for pretraining.
    
    This module controls training, with training_step automatically (and hidden) doing .zero_grad(), .backward() and .step() for optimizer and scheduler.
    
    """

    def __init__(self, conf, model, tokenizer):
        super().__init__()
        self.save_hyperparameters(conf) # saves to self.hparams
        self.automatic_optimization = False # Enable manual optimization
        
        self.loss_fn = get_loss_fn(self.hparams.loss_fn, gamma = self.hparams.fl_gamma)
        self.tokenizer = tokenizer()
        
        self.ablang = model(
            vocab_size = self.hparams.vocab_size,
            hidden_embed_size = self.hparams.hidden_embed_size,
            n_attn_heads = self.hparams.n_attn_heads,
            n_encoder_blocks = self.hparams.n_encoder_blocks,
            padding_tkn = self.hparams.pad_tkn,
            mask_tkn = self.hparams.mask_tkn,
            layer_norm_eps = self.hparams.layer_norm_eps,
            a_fn = self.hparams.a_fn,
            dropout = self.hparams.dropout, 
            use_tkn_dropout = self.hparams.use_tkn_dropout,
            use_moe = self.hparams.use_moe,
        )
        self.ablang.apply(self._init_weights) # Initialize weights
        if self.hparams.path_start_weights != None:
            self.ablang.load_state_dict(
                torch.load(
                    os.path.join(self.hparams.path_start_weights, 'model.pt'), 
                    map_location = torch.device(self.device)
                )
            )
        
        replace_linear_with_kfaclinear(self.ablang)
        self.run_evaluations = Evaluations(self.tokenizer, self.hparams) # Initialize evaluations  
        
        self.validation_step_outputs = []
        
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
        
        return self.ablang(tokens, return_aux_loss=True)

    def training_step(self, dataset, batch_idx):
        tokens, labels = dataset['input'], dataset['labels']
        labels_flat = labels.view(-1) # Ensure labels are flattened

        # Get optimizers
        # In manual optimization, self.optimizers() returns the list of optimizers
        optimizers_list = self.optimizers()
        if not isinstance(optimizers_list, list):
            optimizers_list = [optimizers_list] # Ensure it's a list for iteration

        # Forward pass
        logits, aux_loss = self(tokens) # self() calls forward, returns logits and aux_loss
        main_loss = self.loss_fn(logits.view(-1, self.hparams.vocab_size), labels_flat)
        total_loss = main_loss + aux_loss

        # Log total loss (before accumulation scaling)
        # Matches the condition and name from the original logging for train_loss
        if (batch_idx % self.hparams.accumulate_grad_batches) == 0: 
            self.log("evaluation/train_loss", total_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            
            if (batch_idx % 10) == 0: # Original cache clearing condition was also inside this block
                torch.cuda.empty_cache()

        # Manual backward pass: scale loss for accumulation
        scaled_loss = total_loss / self.hparams.accumulate_grad_batches
        self.manual_backward(scaled_loss)

        # Optimizer step (only if it's the end of an accumulation cycle)
        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            for opt in optimizers_list:
                opt.step()
            for opt in optimizers_list:
                opt.zero_grad()

            # Scheduler step
            # self.lr_schedulers() returns the scheduler instance(s)
            schedulers_list = self.lr_schedulers()
            if schedulers_list:
                if not isinstance(schedulers_list, list):
                    schedulers_list = [schedulers_list] # Ensure it's a list
                for sch in schedulers_list:
                    sch.step()
        
        del tokens
        del labels, labels_flat
        del dataset
        
        return {"loss": total_loss.detach()}
    
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
        
        self.validation_step_outputs.append({
            'val_loss': loss, 'perplexity':perplexity, 
            'val_loss_h': loss_heavy, 'perplexity_h':perplexity_heavy,
            'val_loss_l': loss_light, 'perplexity_l':perplexity_light,
        })
        
        del heavy
        del light
        
        return {'val_loss': loss}
    
    
    def on_validation_epoch_end(self): # Updated once when validation is called
                
        perplexity = torch.stack([x['perplexity'] for x in self.validation_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity"].log(float(perplexity.item()))
        
        val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss"].log(float(val_loss.item()))
        
        perplexity = torch.stack([x['perplexity_h'] for x in self.validation_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity_h"].log(float(perplexity.item()))
        
        val_loss = torch.stack([x['val_loss_h'] for x in self.validation_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss_h"].log(float(val_loss.item()))
        
        perplexity = torch.stack([x['perplexity_l'] for x in self.validation_step_outputs]).mean() # mean across each batch
        self.logger.experiment["evaluation/perplexity_l"].log(float(perplexity.item()))
        
        val_loss = torch.stack([x['val_loss_l'] for x in self.validation_step_outputs]).mean()
        self.logger.experiment["evaluation/val_loss_l"].log(float(val_loss.item()))
        
        self.run_evaluations(self)
        self.validation_step_outputs.clear()  # free memory
        

    
    def configure_optimizers(self):
        """
        Initialization of the optimizer and scheduler used for training
        """
        model = self.ablang
        no_decay = ["bias", "LayerNorm.weight"]
        
        kfac_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            is_kfac_layer = False
            # Check if the parameter belongs to a KFACLinear layer
            # This requires iterating up the module hierarchy if modules are nested
            # For simplicity, assuming KFACLinear layers are direct children or easily identifiable
            # A more robust way would be to check the type of the module 'param' belongs to.
            # This part might need refinement based on the exact model structure.
            # We'll find the module the parameter 'param' belongs to
            module_name_parts = name.split('.')
            parent_module = model
            try:
                for part in module_name_parts[:-1]: # Navigate to the parent module
                    parent_module = getattr(parent_module, part)
                if isinstance(parent_module, KFACLinear):
                    is_kfac_layer = True
            except AttributeError:
                # This can happen if the parameter is directly in the top-level model,
                # or if the naming convention doesn't match module structure.
                pass


            if is_kfac_layer:
                kfac_params.append(param)
            else:
                if not any(nd in name for nd in no_decay):
                    adamw_params.append({
                        "params": [param],
                        "weight_decay": self.hparams.weight_decay,
                    })
                else:
                    adamw_params.append({
                        "params": [param],
                        "weight_decay": 0.0,
                    })

        optimizers = []
        schedulers = []

        if kfac_params:
            # These should be configured appropriately for distributed training.
            # rank = self.global_rank if hasattr(self, 'global_rank') else 0
            # world_size = self.world_size if hasattr(self, 'world_size') else 1
            
            # KOAP init updated based on refactored version
            koap_optimizer = KOAP(kfac_params, 
                                  model, # KOAP needs the model to find KFACLinear layers
                                  lr=self.hparams.learning_rate,
                                  betas=self.hparams.adam_betas, # Adam betas for moment estimates
                                  eps=self.hparams.adam_epsilon,
                                  damping=self.hparams.kfac_damping if hasattr(self.hparams, 'kfac_damping') else 0.01,
                                  precondition_warmup=self.hparams.kfac_precondition_warmup if hasattr(self.hparams, 'kfac_precondition_warmup') else 10,
                                  precondition_frequency=self.hparams.kfac_precondition_frequency if hasattr(self.hparams, 'kfac_precondition_frequency') else 10,
                                  top=self.hparams.kfac_top if hasattr(self.hparams, 'kfac_top') else 1.0,
                                  mode=self.hparams.kfac_mode if hasattr(self.hparams, 'kfac_mode') else 'eigh'
                                  )
            optimizers.append(koap_optimizer)
            
            koap_scheduler = get_cosine_schedule_with_warmup(koap_optimizer,
                                                            num_warmup_steps=self.hparams.warmup_steps,
                                                            num_training_steps=self.hparams.num_training_steps
                                                            )
            schedulers.append({"scheduler": koap_scheduler, "interval": "step", "frequency": 1})

        if adamw_params:
            adamw_optimizer = torch.optim.AdamW(adamw_params,
                                             lr=self.hparams.learning_rate,
                                             betas=self.hparams.adam_betas,
                                             eps=self.hparams.adam_epsilon
                                             # weight_decay is already set in optimizer_grouped_parameters
                                             )
            optimizers.append(adamw_optimizer)
            adamw_scheduler = get_cosine_schedule_with_warmup(adamw_optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.hparams.num_training_steps
                                                       )
            schedulers.append({"scheduler": adamw_scheduler, "interval": "step", "frequency": 1})
        
        return optimizers, schedulers

