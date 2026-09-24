from tokenizers import Tokenizer,pre_tokenizers,models,trainers

import torch

def build_tokenizer(texts,vocab_size=8000):
    tok = Tokenizer(model=models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts,trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=["[PAD]","[UNK]"]))
    return tok

def encode_batch(tokenizer,texts,max_len=64):
    row = []
    for e in tokenizer.encode_batch(texts):
        ids = e.ids
        ids = ids[:max_len]
        ids = ids + [0] * (max_len-len(ids))
        row.append(ids)
    return torch.tensor(row,dtype=torch.long)