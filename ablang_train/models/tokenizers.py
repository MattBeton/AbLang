import json
import torch

from .vocab import ablang_vocab

class ABtokenizer:
    """
    Tokenizer for proteins (focus on handling antibodies). Both aa to token and token to aa.
    """
    
    def __init__(self, vocab_dir=None):
        self.set_vocabs(vocab_dir)
        
    def __call__(self, sequence_list, encode=True, pad=False, add_extra_tkns=True, device='cpu'):
        
        sequence_list = [sequence_list] if isinstance(sequence_list, str) else sequence_list
        
        if encode: 
            data = [self.encode(seq, add_extra_tkns = add_extra_tkns, device = device) for seq in sequence_list]
            if pad: return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=self.pad_token)
            else: return data
        
        else: return [self.decode(tokenized_seq) for tokenized_seq in sequence_list]
    
    def set_vocabs(self, vocab_dir):
        
        if vocab_dir:
            with open(vocab_dir, encoding="utf-8") as vocab_handle:
                self.vocab_to_token=json.load(vocab_handle)
        else:
            self.vocab_to_token = ablang_vocab
            
        self.vocab_to_aa = {v: k for k, v in self.vocab_to_token.items()}
        self.pad_token = self.vocab_to_token['-']
        self.start_token = self.vocab_to_token['<']
        self.end_token = self.vocab_to_token['>']
        self.sep_token = self.vocab_to_token['|']
     
    def encode(self, sequence, add_extra_tkns=True, device='cpu'):
        
        if add_extra_tkns:
            tokenized_seq = [self.vocab_to_token["<"]]+[self.vocab_to_token[resn] for resn in sequence]+[self.vocab_to_token[">"]]
        else:
            tokenized_seq = [self.vocab_to_token[resn] for resn in sequence]
        
        return torch.tensor(tokenized_seq, dtype=torch.long, device=device)
    
    def decode(self, tokenized_seq):
        
        if torch.is_tensor(tokenized_seq): tokenized_seq = tokenized_seq.cpu().numpy()

        return ''.join([self.vocab_to_aa[token] for token in tokenized_seq])
    

    
