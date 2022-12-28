import torch

from ablang_train.train_utils.datacollators import ABcollator

class LossAndPerplexity:
    
    def __init__(self, tokenizer, hparams):
        
        self.mask_tkn = hparams.mask_tkn
        self.vocab_size = hparams.vocab_size
        self.tokenizer = tokenizer
        
        self.fast_collater = ABcollator(
            tokenizer, 
            pad_tkn = hparams.pad_tkn,
            start_tkn = hparams.start_tkn,
            end_tkn = hparams.end_tkn,
            sep_tkn = hparams.sep_tkn,
            mask_tkn = hparams.mask_tkn,
            mask_percent=hparams.mask_percent,
            cdr3_focus=1.,
            mask_technique='random',
        )
        
    def calculate_perplexity_fast(self, trainer, sequences):
        """
        Fast implementation of perplexity. This version requires only a single forward pass, 
        but is stochastic and might focus on easy regions by chance.
        
        Idea taken from ESM-2 paper.
        
        It will start near infinity and then decrease to below 1-2. 
        
        Preferable for during training estimate of perplexity.
        """

        model = trainer.ablang

        tokenized_seqs = self.fast_collater(sequences)

        logits = model(tokenized_seqs['input'].to(trainer.device))
        loss = trainer.loss_fn(logits.view(-1, self.vocab_size), tokenized_seqs['labels'].to(trainer.device))
        
        return loss, torch.exp(loss)
    
    
    def calculate_perplexity_slow(self, trainer, sequences):
        """
        Slow implementation of perplexity. This version requires, instead of a single pass per sequence, 
        a pass for each residue up to the length of the longest sequence in the batch per sequence.
        This version is ~192 times slower than the fast version, but is more precise and deterministic.
        
        Idea taken from ESM-2 paper.
        
        Preferable for generating paper perplexity values.
        
        """
        
        model = trainer.ablang
        
        tokenized_seqs = self.tokenizer(sequences, pad=True, add_extra_tkns=False, device=trainer.device)

        repeat_tokenized_seqs = tokenized_seqs.repeat(tokenized_seqs.size(-1), 1)
        diagonal_mask = torch.ones(tokenized_seqs.size(-1)-1).diag(1).repeat(tokenized_seqs.size(0), 1)

        masked_seqs = repeat_tokenized_seqs.masked_fill(diagonal_mask == 1, self.mask_tkn)

        labels = repeat_tokenized_seqs.masked_fill(masked_seqs != self.mask_tkn, -100).where(
            (repeat_tokenized_seqs!=22) * (repeat_tokenized_seqs!=21) * (repeat_tokenized_seqs!=25), torch.tensor(-100)
        )

        logits = model(masked_seqs)

        loss = trainer.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return loss, torch.exp(loss)