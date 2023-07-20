import torch
import math

import numpy as np


class ABcollator():
    """
    This collator creates; 
    1. masked input data
    2. labels with only masks visible
    
    Padded tokens are also masked.
    """
    def __init__(
        self, 
        tokenizer, 
        pad_tkn=21, 
        start_tkn=0, 
        end_tkn=22,
        sep_tkn=25, 
        mask_tkn=23,
        mask_percent=.15, 
        mask_variable=False, 
        cdr3_focus=1, 
        mask_technique='random',
        change_percent=.1,
        leave_percent=.1,
    ):
        
        self.tokenizer = tokenizer
        self.pad_tkn = pad_tkn 
        self.start_tkn = start_tkn 
        self.end_tkn = end_tkn 
        self.sep_tkn = sep_tkn 
        self.mask_tkn = mask_tkn
        self.mask_percent = mask_percent
        self.mask_variable = mask_variable
        self.cdr3_focus = cdr3_focus
        self.mask_technique = mask_technique
        self.change_percent = change_percent
        self.leave_percent = leave_percent
        
        
    def get_mask_arguments(self, tkn_sequences):
        
        mask_num = int(tkn_sequences.shape[1] * self.mask_percent)
        if self.mask_variable: mask_num = np.random.randint(10, mask_num + 5, size=None)
        
        change_percent = self.change_percent
        if self.change_percent == -1: change_percent = np.random.choice([.1, .2, .4, .6, .8], size=None)
        
        
        if self.mask_technique == 'mix':
            mask_technique = np.random.choice(['random', 'span_long', 'span_short'], p = (1/3, 1/3,1/3), size=None) # Removed short_span because of bugs
            if mask_technique == 'span_short':
                mask_num = 50
            return mask_num, mask_technique, change_percent
        else:
            if self.mask_technique == 'span_short':
                mask_num = 50
            return mask_num, self.mask_technique, change_percent
        
        
    def __call__(self, batch):
        
        tkn_sequences = self.tokenizer(batch, w_extra_tkns=False, pad=True)

        mask_num, mask_technique, change_percent = self.get_mask_arguments(tkn_sequences)
        #print(mask_num, mask_technique, change_percent)
        masked_sequences, target_tkns = generate_masked_sequences(
            tkn_sequences, 
            pad_tkn = self.pad_tkn,
            start_tkn = self.start_tkn,
            end_tkn = self.end_tkn,
            sep_tkn = self.sep_tkn,
            mask_tkn = self.mask_tkn,
            mask_num=mask_num, 
            cdr3_focus=self.cdr3_focus,
            mask_technique = mask_technique,
            change_percent = change_percent,
            leave_percent = self.leave_percent,
        )
        
        return {'input':masked_sequences, 'labels':target_tkns.view(-1), 'sequences':batch}


def generate_masked_sequences(
    tkn_sequences, 
    pad_tkn=21, 
    start_tkn=0, 
    end_tkn=22,
    sep_tkn=25, 
    mask_tkn=23, 
    mask_num=15, 
    cdr3_focus=1,
    mask_technique='random',
    change_percent=.1,
    leave_percent=.1,
):
    """
    Same as create_BERT_data, but also keeps start and stop.
    """
    stop_start_mask = ((tkn_sequences == start_tkn) | (tkn_sequences == sep_tkn) | (tkn_sequences == end_tkn))
    attention_mask = tkn_sequences.eq(pad_tkn)
    
    if mask_num == 0: # This is for validation cases
        masked_sequences = tkn_sequences.clone()
        target_sequences = tkn_sequences.clone()
        target_sequences[attention_mask] = -100
        return masked_sequences, target_sequences
    
    allowed_mask = get_allowed_mask(attention_mask, stop_start_mask, mask_technique, mask_num)

    idx_change, idx_leave, idx_mask = get_indexes(
        allowed_mask, 
        mask_num=mask_num, 
        change_percent=change_percent, 
        leave_percent=leave_percent, 
        cdr3_focus=cdr3_focus, 
        mask_technique=mask_technique
    )
    
    masked_sequences = tkn_sequences.clone()
    masked_sequences.scatter_(1, idx_change, torch.randint(1, 21, masked_sequences.shape, device=masked_sequences.device)) # randomly changes idx_change in the data 
    masked_sequences.scatter_(1, idx_mask, mask_tkn) # change idx_mask inputs to <mask>
    
    target_sequences = tkn_sequences.clone()
    stop_start_mask.scatter_(1, idx_mask, 1)
    stop_start_mask.scatter_(1, idx_change, 1)
    stop_start_mask.scatter_(1, idx_leave, 1)
    target_sequences[~stop_start_mask.long().bool()] = -100
    target_sequences[(tkn_sequences == pad_tkn)] = -100
    
    return masked_sequences, target_sequences


def get_allowed_mask(attention_mask, stop_start_mask, mask_technique, mask_num):
    
    base_mask = (~(attention_mask + stop_start_mask)).float()
    
    if mask_technique == 'random':
        return base_mask
    
    elif 'span' in mask_technique:
        """
        Removes the end possible masks, so get_indexes doesn't mask things outside of the sequences.
        """
        return re_adjust_matrix(base_mask, attention_mask, mask_num)
        
        
def re_adjust_matrix(base_mask, attention_mask, mask_num):

    idx = torch.arange(attention_mask.shape[1], 0, -1)
    indices = torch.argmax(attention_mask * idx, 1, keepdim=True)

    for test_idx in indices.reshape(-1):
        base_mask[:,test_idx - mask_num - 1:attention_mask.shape[1]] = 0    
    
    return base_mask
    
    
def correct_idxs(idx, max_len):
    
    if idx.max() > max_len:
        idx = idx[:,:-1]
        return correct_idxs(idx, max_len)
    else:
        return idx

    
def get_indexes(
    allowed_mask, 
    mask_num, 
    change_percent=.1, 
    leave_percent=.1, 
    cdr3_focus = 1,
    mask_technique = 'random'
):
    #allowed_mask[:, 106:118] *= cdr3_focus # Changes the chance of residues in the CDR3 getting masked. It's 106 and 118 because the start token is present.
    
    if mask_technique == 'random':
        idx = torch.multinomial(allowed_mask.float(), num_samples=mask_num, replacement=False)
    
    elif mask_technique == 'span_long':

        start_idx = torch.multinomial(allowed_mask.float(), num_samples = 1, replacement = False).repeat(1, mask_num)
        step_idx = torch.linspace(0, mask_num-1, steps = mask_num, dtype = int).repeat(allowed_mask.shape[0], 1)
        idx = start_idx + step_idx
        idx = correct_idxs(idx, allowed_mask.shape[1]) 
        
    elif mask_technique == 'span_short':        
        span_lengths = np.random.choice([2, 3, 4], size=(5))
        span_separation_lengths = torch.normal(mean=15, std=6, size=(10,)).int()
        span_separation_lengths = torch.where(torch.where(span_separation_lengths < 1, 1, span_separation_lengths) > 15, 15, span_separation_lengths)

        start_idx = torch.multinomial(allowed_mask.float(), num_samples = 1, replacement = False).repeat(1, mask_num)
        step_idx = torch.linspace(0, mask_num-1, steps = mask_num, dtype = int).repeat(allowed_mask.shape[0], 1)
        idx = start_idx + step_idx

        start_idx = 0
        many_span_idx = []
        for lengths, separation in zip(span_lengths, span_separation_lengths):
            many_span_idx.append(idx[:,start_idx:start_idx+lengths])
            start_idx += lengths + separation

        idx = torch.concatenate(many_span_idx, axis=1)
        idx = correct_idxs(idx, allowed_mask.shape[1])    
        
    n_change = max(int(idx.shape[1]*change_percent), 1)
    n_leave  = max(int(idx.shape[1]*leave_percent ), 0)

    return torch.split(idx, split_size_or_sections = [n_change, n_leave, max(idx.shape[-1] - (n_change + n_leave), 0)], dim = 1)
