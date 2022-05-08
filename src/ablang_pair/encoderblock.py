import torch
import math
import torch.nn.functional as F
import einops


class TransformerEncoder(torch.nn.Module):
    """
    Single Transformer Encoder.
    
    An Transformer Encoder consists of a MultiHeadAttention and a IntermediateLayer.
    """
    def __init__(self, hparams):
        super().__init__()
        
        assert hparams.representation_size % hparams.num_attention_heads == 0, \
        "Embedding dimension must be 0 modulo number of heads." 
        
        self.mha_attention = Attention(
            input_dim = hparams.representation_size, 
            embed_dim = hparams.representation_size, 
            num_heads = hparams.num_attention_heads,
            attention_dropout_prob = hparams.attention_dropout_prob
        )
        
        self.intermediate_layer = torch.nn.Sequential(
            torch.nn.Linear(hparams.representation_size, hparams.intermediate_size),
            torch.nn.GELU(),
            torch.nn.Dropout(hparams.representation_dropout_prob),
            torch.nn.Linear(hparams.intermediate_size, hparams.representation_size),
            torch.nn.Dropout(hparams.representation_dropout_prob),
        )
        
        self.dropout = torch.nn.Dropout(hparams.representation_dropout_prob)
        self.layer_norm_1 = torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)
        self.layer_norm_2 = torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)
        self.layer_norm_3 = torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps)
        
    def forward(self, representations, attention_mask=None, output_attentions=False):
        
        representations = self.layer_norm_1(representations) # DERIVED FROM ESM-1b?
        
        mha_out, attn_weights = self.mha_attention(representations, 
                                                 mask=attention_mask, 
                                                 return_attention=output_attentions
                                                )
        representations = self.layer_norm_2(representations + self.dropout(mha_out)) # RESIDUAL BLOCK EFFECT
        
        iml_out = self.intermediate_layer(representations) 
        representations = self.layer_norm_3(representations + iml_out) # RESIDUAL BLOCK EFFECT
        
        return representations, attn_weights

    
class Attention(torch.nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, attention_dropout_prob=None):
        super().__init__()
        
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        
    def _attention(self, q, k, v, mask=None):
        
        d_k = q.size()[-1]
        
        # scaled_dot_product_attention
        scores = torch.matmul(q, k.transpose(-2, -1)) 
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            mask = einops.rearrange(mask, 'b_size (h1 h2 seq_len) -> b_size h1 h2 seq_len', h1=1, h2=1)
            _MASKING_VALUE = -1e+9 if scores.dtype == torch.float32 else -1e+4
            scores = scores.masked_fill(mask, _MASKING_VALUE)

        attention = F.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)

        output = torch.matmul(attention, v)
        return output, attention

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        values, attention = self._attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.out_proj(values)

        if return_attention:
            return o, attention
        else:
            return o, None
    
    