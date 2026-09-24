import torch
import torch.nn as nn

LABELS = ["negative","positive"]
NUM_CLASSES = 2

class SentimentClassifier(nn.Module):
    def __init__(self,vocab_size,d_model=128,nhead=4,num_layers=2,dim_feedforward=256,num_classes=NUM_CLASSES,max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.pos = nn.Embedding(max_len,d_model)
        layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,num_layers)
        self.head = nn.Linear(d_model,num_classes)

    def forward(self,x):
        pad_mask = x == 0
        L = x.size(1)
        pos = torch.arange(L,device=x.device)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h,src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(h)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
