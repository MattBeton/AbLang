import torch
from torch import nn
import torch.nn.functional as F

from .extra_fns import ACT2FN
from .encoderblocks import EncoderBlocks
from .embedding import AbEmbeddings


class AbLang(torch.nn.Module):
    """Pretraining model includes Abrep and the head model used for training."""
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        self.AbRep = AbRep(self.hparams)       
        self.AbHead = AbHead(self.hparams, self.AbRep.AbEmbeddings.AAEmbeddings.weight)
        
    def forward(self, x, attention_mask=None):
        
        representations = self.AbRep(x, attention_mask)
        
        likelihoods = self.AbHead(representations.last_hidden_states)
        
        return likelihoods
    
    def get_aa_embeddings(self):
        "This function is used to extract the trained aa_embeddings."
        return self.AbRep.AbEmbeddings.aa_embeddings#().weight.detach()

    
class AbRep(torch.nn.Module):
    """This is the AbRep model."""
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        self.AbEmbeddings = AbEmbeddings(self.hparams)    
        self.EncoderBlocks = EncoderBlocks(self.hparams)
        
        self.init_weights()
        
    def forward(self, src, attention_mask=None, output_attentions=False):
        
        attention_mask = torch.zeros(*src.shape, device=src.device).masked_fill(src == self.hparams.pad_tkn, 1)

        src = self.AbEmbeddings(src)
        
        output = self.EncoderBlocks(src, attention_mask=attention_mask, output_attentions=output_attentions)
        
        return output
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
    

class AbHead(torch.nn.Module):
    """Head model for masked sequence prediction."""
    
    def __init__(self, hparams, weights):
        super().__init__()

        self.dense = nn.Linear(hparams.representation_size, hparams.representation_size)
        self.activation = ACT2FN[hparams.hidden_act]
        self.layer_norm = nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)

        self.weight = weights
        self.bias = nn.Parameter(torch.zeros(hparams.vocab_size))

        

    def forward(self, features, **kwargs):
        x = self.dense(features)

        x = self.activation(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        #x = self.decoder(x)
        x = F.linear(x, self.weight) + self.bias
        
        return x
