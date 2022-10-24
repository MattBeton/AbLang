from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .encoderblock import TransformerEncoder
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
        return self.AbRep.AbEmbeddings.aa_embeddings

    
class AbRep(torch.nn.Module):
    """This is the AbRep model."""
    
    def __init__(self, hparams):
        super().__init__()
        self.pad_tkn = hparams.pad_tkn
        
        self.AbEmbeddings = AbEmbeddings(hparams)    
        self.EncoderBlocks = torch.nn.ModuleList([TransformerEncoder(hparams) for _ in range(hparams.num_encoder_blocks)])
        self.LayerNorm = torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)
        
    def additional():
        
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []
        
    def forward(self, features, attention_mask=None, output_attentions=False, output_representations=False):
        
        attention_mask = (features == self.pad_tkn) # Needs to be here for when you eval

        representation = self.AbEmbeddings(features)
        
        all_representations = () if output_representations else None
        all_self_attentions = () if output_attentions else None

        
        for EncoderBlock in self.EncoderBlocks:  
            representation, attentions = EncoderBlock(representation, attention_mask, output_attentions)
            
            if output_representations: 
                all_representations = all_representations + (representation,) # Takes out each hidden states after each EncoderBlock
            
            if output_attentions: 
                all_self_attentions = all_self_attentions + (attentions,) # Takes out attention layers for analysis
           
        representation = self.LayerNorm(representation)

        return AbRepOutput(last_hidden_states=representation, all_hidden_states=all_representations, attentions=all_self_attentions)
    

class AbHead(torch.nn.Module):
    """Head model for masked sequence prediction."""
    
    def __init__(self, hparams, weights):
        super().__init__()
        
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hparams.representation_size, hparams.representation_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps),
        )

        self.weight = weights
        self.bias = torch.nn.Parameter(torch.zeros(hparams.vocab_size))   
        self.final_layernorm = torch.nn.LayerNorm(hparams.vocab_size,
                                                  eps=hparams.layer_norm_eps)

    def forward(self, features, **kwargs):
        
        x = self.ff(features)

        x = F.linear(x, self.weight) + self.bias
        
        return self.final_layernorm(x)

    
@dataclass
class AbRepOutput():
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None