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
    def __init__(self, tokenizer, pad_to_mask=21, mask_percent=.15, mask_variable=False, cdr3_focus=1):
        self.tokenizer = tokenizer
        self.pad_to_mask = pad_to_mask 
        self.mask_percent = mask_percent
        self.mask_variable = mask_variable
        self.cdr3_focus = cdr3_focus
        
    def __call__(self, batch):
        
        data = self.tokenizer(batch, pad=True)
        
        targets = data.clone()

        if self.mask_variable:
            mask_percent = np.random.uniform(low=0.0, high=self.mask_percent, size=None)
        else:
            mask_percent = self.mask_percent

        new_data, data_mask, new_targets = create_stop_start_data(data, 
                                                                  pad_token=self.pad_to_mask, 
                                                                  start_token=0, 
                                                                  stop_token=22, 
                                                                  mask_percent=mask_percent, 
                                                                  cdr3_focus=self.cdr3_focus
                                                                 )
        
        return {'input':new_data, 'labels':new_targets.view(-1), 'attention_mask':data_mask}


def create_stop_start_data(data, 
                           pad_token=21, 
                           start_token=0, 
                           stop_token=22, 
                           mask_percent=.15, 
                           cdr3_focus=1):
    """
    Same as create_BERT_data, but also keeps start and stop.
    """
    stop_start_mask = ((data == start_token) | (data == stop_token) | (data == 24))
    attention_mask = (data == pad_token)
    
    sequence_mask = (~(attention_mask + stop_start_mask)).float()
    
    if not mask_percent > 0: # 0% MASKING JUST REMOVES THE FEATURE - STILL MAKES AWAY PADDING
        changed_data = data.clone()
        new_targets = data.clone()
        new_targets[attention_mask] = -100
        
        return changed_data, attention_mask, new_targets

    idx_change, _, idx_mask = get_indexes(sequence_mask, mask_percent=mask_percent, change_percent=.1, leave_percent=.1, cdr3_focus=cdr3_focus)
    
    changed_data = data.clone()
    changed_data.scatter_(1, idx_change, torch.randint(1, 20, changed_data.shape, device=data.device)) # randomly changes idx_change in the data 
    changed_data.scatter_(1, idx_mask, 23) # change idx_mask inputs to <mask>
    
    new_targets = data.clone()
    target_mask = stop_start_mask.clone()
    target_mask.scatter_(1, idx_mask, 1)
    new_targets[~target_mask.long().bool()] = -100
    
    return changed_data, attention_mask, new_targets


def get_indexes(matrix_to_mask, mask_percent, change_percent=.1, leave_percent=.1, cdr3_focus = 1): 
    
    matrix_to_mask[:, 106:118] *= cdr3_focus # Changes the chance of residues in the CDR3 getting masked. It's 106 and 118 because the start token is present.
    
    idx = torch.multinomial(matrix_to_mask.float(), num_samples=int(math.ceil(mask_percent*matrix_to_mask.shape[1])), replacement=False)

    n_change = int(idx.shape[1]*change_percent)
    n_leave = int(idx.shape[1]*leave_percent)
    
    return torch.split(idx, split_size_or_sections=[n_change, n_leave, idx.shape[-1] - (n_change +n_leave)], dim=1)
