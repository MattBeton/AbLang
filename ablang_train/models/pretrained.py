import numpy as np
import torch

from .load_model import load_model
from .pretrained_utils.restoration import RestoreAntibody
from .pretrained_utils.encodings import AbEncoding


class pretrained:
    """
    Initializes AbLang for heavy or light chains.    
    """
    
    def __init__(self, model_to_use="download", chain="heavy", random_init=False, ncpu=1, device='cpu'):
        super().__init__()
        
        self.used_device = torch.device(device)
        
        self.AbLang, self.tokenizer, self.hparams = load_model(model_to_use)
        self.AbLang.eval() # Default 
        
        #self.hparams = hparams
        self.AbRep = self.AbLang.AbRep
        
        self.ncpu = ncpu
        self.spread = 11 # Based on get_spread_sequences function
        if chain == 'heavy':
            self.max_position = 128
        else:
            self.max_position = 127
            
        self.restore_antibody = RestoreAntibody(self.AbLang, self.tokenizer, self.spread, self.max_position, self.used_device, ncpu)
        self.encode_antibody = AbEncoding(self.AbLang, self.tokenizer, self.spread, self.max_position, self.used_device, ncpu)
        
    def freeze(self):
        self.AbLang.eval()
        
    def unfreeze(self):
        self.AbLang.train()
        
    def __call__(self, sequence, mode='seqcoding', align=False, chain ='H', chunk_size=50):
        """
        Mode: sequence, residue, restore or likelihood.
        """
        if not mode in ['rescoding', 'seqcoding', 'restore', 'likelihood']:
            raise SyntaxError("Given mode doesn't exist.")
        
        if isinstance(sequence, str): sequence = [sequence]
        
        
        aList = []
        for sequence_part in [sequence[x:x+chunk_size] for x in range(0, len(sequence), chunk_size)]:
            aList.append(getattr(self, mode)(sequence_part, align, chain))
        
        if mode == 'rescoding':
            if align==True:
                return aList
            
            return sum(aList, [])
        
        return np.concatenate(aList)
    
    def seqcoding(self, seqs, align=False, chain ='H'):
        """
        Sequence specific representations
        """
        
        return self.encode_antibody.get_seq_coding(seqs)
    
    def restore(self, seqs, align=False, chain ='H'):
        """
        Restore sequences
        """

        return self.restore_antibody.restore(seqs, align=align)
    
    def likelihood(self, seqs, align=False, chain ='H'):
        """
        Possible Mutations
        """
        
        tokens = self.tokenizer(seqs, pad=True, device=self.used_device)
        
        predictions = self.AbLang(tokens)[:,:,1:21]
        
        if torch.is_tensor(predictions): predictions = predictions.cpu().detach().numpy()

        return predictions
    
    def rescoding(self, seqs, align=False, chain = 'H'):
        """
        Residue specific representations.
        """
           
        return self.encode_antibody.get_res_coding(seqs, align=align, chain = chain)
        