from .eval_plots import plot_aa_embeddings
from .seq_restoration import log_restoring_sequence
from .loss_and_perplexity import LossAndPerplexity

class Evaluations:
    
    def __init__(self, tokenizer, hparams):
        
        self.tokenizer = tokenizer
        self.hparams = hparams
        
        self.loss_n_perplexity = LossAndPerplexity(tokenizer, hparams)
    
    
    def __call__(self, trainer, val_step_outputs=''):
                
        #plot_aa_embeddings(trainer)
        log_restoring_sequence(trainer)
        