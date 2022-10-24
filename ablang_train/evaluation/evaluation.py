from .embeddings import plot_Ab_embeddings, plot_Ap_embeddings
#from .loss_test import log_valuation_loss, log_restoring_sequence


class Evaluations:
    
    def __init__(self):
        
        pass
    
    
    def __call__(self, trainer, val_step_outputs=''):
        
        
        #log_valuation_loss(trainer, val_step_outputs)
        #log_restoring_sequence(trainer)
        
        plot_Ab_embeddings(trainer)
        plot_Ap_embeddings(trainer)
        