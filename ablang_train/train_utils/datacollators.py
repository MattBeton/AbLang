import torch
import math

import numpy as np


class ABcollator():
    """
    This collator creates; 
    1. masked input data
    2. labels with only masks visible
    3. attention_mask based on masking
    
    Padded tokens are also masked.
    """
    def __init__(self, tokenizer, 
                 pad_tkn=21, 
                 cls_tkn=0, 
                 sep_tkn=22, 
                 mask_tkn=23,
                 mask_percent=.15, 
                 mask_variable=False, 
                 cdr3_focus=1, 
                 mask_technique='random'):
        
        self.tokenizer = tokenizer
        self.pad_tkn = pad_tkn 
        self.cls_tkn = cls_tkn 
        self.sep_tkn = sep_tkn 
        self.mask_tkn = mask_tkn
        self.mask_percent = mask_percent
        self.mask_variable = mask_variable
        self.cdr3_focus = cdr3_focus
        self.mask_technique = mask_technique
        
    def get_mask_arguments(self, tkn_sequences):
        
        mask_num = int(tkn_sequences.shape[0] * self.mask_percent)
        
        if self.mask_variable:
            mask_num = np.random.randint(1, mask_num, size=None)
            
        if self.mask_technique == 'mix':
            mask_technique = np.random.choice(['random','connected'], size=None)
        else:
            mask_technique = self.mask_technique
            
        return mask_num, mask_technique
        
        
    def __call__(self, batch):
        
        tkn_sequences = self.tokenizer(batch, pad=True)

        mask_num, mask_technique = self.get_mask_arguments(tkn_sequences)

        masked_sequences, target_tkns = generate_masked_sequences(tkn_sequences, 
                                                                  pad_tkn=self.pad_tkn, 
                                                                  cls_tkn=self.cls_tkn, 
                                                                  sep_tkn=self.sep_tkn, 
                                                                  mask_tkn=self.mask_tkn,
                                                                  mask_num=mask_num, 
                                                                  cdr3_focus=self.cdr3_focus,
                                                                  mask_technique = mask_technique
                                                                 )
        
        return {'input':masked_sequences, 'labels':target_tkns.view(-1), 'sequences':batch}


def generate_masked_sequences(tkn_sequences, 
                           pad_tkn=21, 
                           cls_tkn=0, 
                           sep_tkn=22, 
                           mask_tkn=23, 
                           mask_num=15, 
                           cdr3_focus=1,
                           mask_technique='random'
                          ):
    """
    Same as create_BERT_data, but also keeps start and stop.
    """
    stop_start_mask = ((tkn_sequences == cls_tkn) | (tkn_sequences == sep_tkn))
    attention_mask = tkn_sequences.eq(pad_tkn)
    
    allowed_mask = get_allowed_mask(attention_mask, stop_start_mask, mask_technique, mask_num)
    
    
    if mask_num == 0: # This is for validation cases
        masked_sequences = tkn_sequences.clone()
        tkn_sequences[attention_mask] = -100
        return masked_sequences, tkn_sequences

    idx_change, _, idx_mask = get_indexes(
        allowed_mask, 
        mask_num=mask_num, 
        change_percent=.1, 
        leave_percent=.1, 
        cdr3_focus=cdr3_focus, 
        mask_technique=mask_technique
    )
    
    masked_sequences = tkn_sequences.clone()
    masked_sequences.scatter_(1, idx_change, torch.randint(1, 20, masked_sequences.shape, device=masked_sequences.device)) # randomly changes idx_change in the data 
    masked_sequences.scatter_(1, idx_mask, mask_tkn) # change idx_mask inputs to <mask>
    
    stop_start_mask.scatter_(1, idx_mask, 1)
    tkn_sequences[~stop_start_mask.long().bool()] = -100
    
    return masked_sequences, tkn_sequences


def get_allowed_mask(attention_mask, stop_start_mask, mask_technique, mask_num):
    
    base_mask = (~(attention_mask + stop_start_mask)).float().clone()
    
    if mask_technique == 'random':
        return base_mask
    
    elif mask_technique == 'connected':
        """
        Removes the end possible masks, so get_indexes doesn't mask things outside of the sequences.
        """
        
        return re_adjust_matrix(base_mask, stop_start_mask, mask_num)
        
        
def re_adjust_matrix(matrix, stop_start_mask, adjustment):

    test_idxs = (stop_start_mask.float()==1).nonzero()

    for test_idx in test_idxs:
        matrix[test_idx[0],range(test_idx[1]-adjustment, test_idx[1])] = 0

    return matrix
    

def get_indexes(allowed_mask, 
                mask_num, 
                change_percent=.1, 
                leave_percent=.1, 
                cdr3_focus = 1,
                mask_technique = 'random'
               ): 
    
    allowed_mask[:, 106:118] *= cdr3_focus # Changes the chance of residues in the CDR3 getting masked. It's 106 and 118 because the start token is present.
    
    
    if mask_technique == 'random':
        idx = torch.multinomial(allowed_mask.float(), num_samples=mask_num, replacement=False)
    
    elif mask_technique == 'connected':

        start_idx = torch.multinomial(allowed_mask.float(), num_samples=1, replacement=False).repeat(1, mask_num)
        step_idx = torch.linspace(0, mask_num-1, steps=mask_num, dtype=int).repeat(allowed_mask.shape[0], 1)
        
        idx = start_idx+step_idx

    
    n_change = int(idx.shape[1]*change_percent)
    n_leave = int(idx.shape[1]*leave_percent)

    return torch.split(idx, split_size_or_sections=[n_change, n_leave, idx.shape[-1] - (n_change +n_leave)], dim=1)
