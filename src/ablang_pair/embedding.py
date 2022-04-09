import torch

class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.pad_tkn = hparams.pad_tkn
        
        self.AAEmbeddings = torch.nn.Embedding(hparams.vocab_size, 
                                         hparams.representation_size, 
                                         padding_idx=self.pad_tkn
                                        )
        self.PositionEmbeddings = torch.nn.Embedding(hparams.max_position_embeddings, 
                                               hparams.representation_size, 
                                               padding_idx=0 # here padding_idx is always 0
                                              ) 
        
        self.RegularisationLayer = torch.nn.Sequential(torch.nn.LayerNorm(hparams.representation_size, eps=hparams.layer_norm_eps),
                                         torch.nn.Dropout(hparams.representation_dropout_prob),
                                        )

    def forward(self, src):
        
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids = self.create_position_ids_from_input_ids(src)   
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return self.RegularisationLayer(embeddings)
        
    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(self.pad_tkn).int()
        
        return torch.cumsum(mask, dim=1).long() * mask
    
