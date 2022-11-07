import os
import numpy as np
import torch
import pytorch_lightning as pl
import math

from ablang_train.train_utils import loss_fn, optimizers, schedulers
from ablang_train.evaluation.evaluation import Evaluations


class TrainingFrame(pl.LightningModule):
    """
    Python-Lightning module for pretraining.
    
    This module controls training, with training_step automatically (and hidden) doing .zero_grad(), .backward() and .step() for optimizer and scheduler.
    
    """

    def __init__(self, conf, model, tokenizer):
        super().__init__()
        self.save_hyperparameters(conf) # saves to self.hparams
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer(os.path.join(self.hparams.data_path,'vocab.json'))
        
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
        self.run_evaluations = Evaluations() # Initialize evaluations

    def forward(self, tokens):
        
        return self.ablang(tokens)

    def training_step(self, dataset, batch_idx): 
        
        tokens, labels = dataset['input'], dataset['labels']
        
        output = self(tokens)

        loss = self.loss_fn(output.view(-1, self.hparams.vocab_size), labels)
        
        # Must clear cache at regular interval
        if (batch_idx % self.hparams.accumulate_grad_batches) % 10 == 0: #self.global_step 
            torch.cuda.empty_cache()
            
        # Only log once every global step
        if batch_idx % self.hparams.accumulate_grad_batches == 0:
            self.logger.experiment['evaluation/train_loss'].log(loss)
        
        return {"loss": loss}
    
    def training_step_end(self, batch_parts):
        
        return {'loss': batch_parts['loss'].mean()}
    
    def validation_step(self, dataset, batch_idx): # Updated every step when validation is called
        
        tokens, labels = dataset['input'], dataset['labels']
        
        pp_score = calculate_perplexity(self, self.tokenizer, dataset['sequences'][:10])
        
        output = self(tokens)
        
        loss = self.loss_fn(output.view(-1, self.hparams.vocab_size), labels)
        
        #all_loss = self.loss_fn(output.view(-1, self.hparams.vocab_size), labels, reduce=False).view(output.shape[:-1])[:, 1:]
                
        #fw1_loss = all_loss[:, :27+1].mean()
        #cdr1_loss = all_loss[:, 27:38+1].mean()
        #fw2_loss = all_loss[:, 38:56+1].mean()
        #cdr2_loss = all_loss[:, 56:65+1].mean()
        #fw3_loss = all_loss[:, 65:105+1].mean()
        #cdr3_loss = all_loss[:, 105:117+1].mean()
        #fw4_loss = all_loss[:, 117:].mean()

        return {
            'val_loss': loss, #'fw1_loss':fw1_loss, 'cdr1_loss':cdr1_loss, 'fw2_loss':fw2_loss, 
            #'cdr2_loss':cdr2_loss, 'fw3_loss':fw3_loss, 'cdr3_loss':cdr3_loss, 'fw4_loss':fw4_loss,
            'pp_score':pp_score,
               }
    
    
    def validation_epoch_end(self, val_step_outputs): # Updated once when validation is called
        
        pp_score = torch.stack([x['pp_score'] for x in val_step_outputs]).mean()
        self.logger.experiment["evaluation/pp_score"].log(pp_score)
        
        self.log_valuation_loss(val_step_outputs)
        self.log_restoring_sequence()
        
        self.run_evaluations(self)

    
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
                "monitor": "train_loss",
                "frequency":1,
            },
        }


    def log_valuation_loss(self, val_step_outputs):

        val_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
        #fw1_loss = torch.stack([x['fw1_loss'] for x in val_step_outputs]).mean()
        #cdr1_loss = torch.stack([x['cdr1_loss'] for x in val_step_outputs]).mean()
        #fw2_loss = torch.stack([x['fw2_loss'] for x in val_step_outputs]).mean()
        #cdr2_loss = torch.stack([x['cdr2_loss'] for x in val_step_outputs]).mean()
        #fw3_loss = torch.stack([x['fw3_loss'] for x in val_step_outputs]).mean()
        #cdr3_loss = torch.stack([x['cdr3_loss'] for x in val_step_outputs]).mean()
        #fw4_loss = torch.stack([x['fw4_loss'] for x in val_step_outputs]).mean()

        self.logger.experiment["evaluation/eval_loss"].log(val_loss)
        #self.logger.experiment["evaluation/fw1_loss"].log(fw1_loss)
        #self.logger.experiment["evaluation/cdr1_loss"].log(cdr1_loss)
        #self.logger.experiment["evaluation/fw2_loss"].log(fw2_loss)
        #self.logger.experiment["evaluation/cdr2_loss"].log(cdr2_loss)
        #self.logger.experiment["evaluation/fw3_loss"].log(fw3_loss)
        #self.logger.experiment["evaluation/cdr3_loss"].log(cdr3_loss)
        #self.logger.experiment["evaluation/fw4_loss"].log(fw4_loss)


    def log_restoring_sequence(self):
        testSeq = 'EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS>'
        aaPreds, aaPreds50 = singleSeqValidation(self, self.tokenizer, testSeq=testSeq)       
        self.logger.experiment['evaluation/heavy_reconstruct'].log(aaPreds[0])
        self.logger.experiment['evaluation/heavy_reconstruct_50'].log(aaPreds50[0])

        testSeq = '>DIVMTQTPSTLSASVGDRVTLTCKASQDISYLAWYQQKPGKAPKKLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQQNSNWTFGQGTKVDIK'
        aaPreds, aaPreds50 = singleSeqValidation(self, self.tokenizer, testSeq=testSeq)       
        self.logger.experiment['evaluation/light_reconstruct'].log(aaPreds[0])
        self.logger.experiment['evaluation/light_reconstruct_50'].log(aaPreds50[0])

        testSeq = 'EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS>DIVMTQTPSTLSASVGDRVTLTCKASQDISYLAWYQQKPGKAPKKLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQQNSNWTFGQGTKVDIK'
        aaPreds, aaPreds50 = singleSeqValidation(self, self.tokenizer, testSeq=testSeq)       
        self.logger.experiment['evaluation/paired_reconstruct'].log(aaPreds[0])
        self.logger.experiment['evaluation/paired_reconstruct_50'].log(aaPreds50[0])
        
        
def singleSeqValidation(model, tokenizer, testSeq):
    """
    Small function used to visualize the training by showing how the reconstruction of a given sequence is improved over training.
    """

    tokenPreds = model(tokenizer([testSeq], pad=True, device=model.device))
    
    tokenMAX = torch.max(torch.nn.Softmax(dim=-1)(tokenPreds), -1)

    aaPreds = tokenizer(tokenMAX[1], encode=False)

    unkMatrix = torch.zeros(tokenMAX[0].shape, dtype=torch.long, device=model.device) + 21
    
    aaPreds50 = ['-'.join(tokenizer(torch.where(tokenMAX[0]<=.5, unkMatrix, tokenMAX[1]).detach(), encode=False)[0].split('<unk>'))]

    return aaPreds, aaPreds50


def calculate_perplexity(model, tokenizer, sentences,  mask_token_id=23):
    """
    62.076110763472364 seems to be the start with random weights
    """
    tensor_input = model.tokenizer(sentences, pad=True)
    
    repeat_input = tensor_input.repeat(tensor_input.size(-1), 1)
    mask = torch.ones(tensor_input.size(-1)-1).diag(1).repeat(tensor_input.size(0), 1)

    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)
    
    labels = repeat_input.masked_fill(masked_input != mask_token_id, -100).where((repeat_input!=22) * (repeat_input!=21), torch.tensor(-100))

    output = model(masked_input.to(model.device))
    
    loss = model.loss_fn(output.view(-1, model.hparams.vocab_size), labels.view(-1).to(model.device))
    result = torch.exp(loss)
    return result


