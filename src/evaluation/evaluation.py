from .embeddings import plot_Ab_embeddings, plot_Ap_embeddings



class Evaluations:
    
    def __init__(self):
        
        pass
    
    
    def __call__(self, trainer):
        
        
        plot_Ab_embeddings(trainer)
        plot_Ap_embeddings(trainer)
        