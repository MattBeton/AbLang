import torch
from fairseq.modules.multihead_attention import MultiheadAttention

    
class TransformerEncoder(torch.nn.Module):
    """
    Single Transformer Encoder.
    
    An Transformer Encoder consists of a MultiHeadAttention and a IntermediateLayer.
    """
    def __init__(self, hparams):
        super().__init__()
        
        self.MultiHeadAttention = MultiHeadAttention(hparams)
        
        self.IntermediateLayer = torch.nn.Sequential(
            torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps),
            torch.nn.Linear(hparams.representation_size, hparams.intermediate_size),
            torch.nn.GELU(),
            torch.nn.Dropout(hparams.representation_dropout_prob),
            torch.nn.Linear(hparams.intermediate_size, hparams.representation_size),
            torch.nn.Dropout(hparams.representation_dropout_prob),
        )
        
    def forward(self, representations, attention_mask=None, output_attentions=False):

        afterMHA, attentions = self.MultiHeadAttention(representations, attention_mask, output_attentions=output_attentions)
        
        representations = self.IntermediateLayer(afterMHA) 
        representations = representations + afterMHA # RESIDUAL BLOCK EFFECT
        
        return representations, attentions
    
    
class MultiHeadAttention(torch.nn.Module):
    """
    MultiHeadAttention which can return the weights of the individual heads.
    """
    
    def __init__(self, hparams):
        super().__init__()
                
        self.Attention = MultiheadAttention(hparams.representation_size, 
                                            hparams.num_attention_heads, 
                                            dropout=hparams.attention_dropout_prob, 
                                            self_attention=True)
        
        self.MHADropout = torch.nn.Dropout(hparams.representation_dropout_prob)
        self.MHALayerNorm = torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)
        
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
                
        out = self.MHALayerNorm(hidden_states)
            
        out = out.transpose(0, 1)
        
        # static_kv is only True because there is currently a bug which doesn't return the head weights unaveraged unless its true
        attn_output, attn_weights = self.Attention(out, 
                                                   out, 
                                                   out, 
                                                   key_padding_mask=attention_mask, 
                                                   static_kv=True, 
                                                   need_weights=output_attentions, 
                                                   need_head_weights=output_attentions)

        out = attn_output.transpose(0, 1)
        
        out = self.MHADropout(out)
        out = out + hidden_states # HIDDEN_STATES ARE ADDED FOR RESIDUAL BLOCK EFFECT
        
        return out, attn_weights
