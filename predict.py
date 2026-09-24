import torch
from tokens import encode_batch
from model import LABELS

def predict(model,tokenizer,text):
    x = encode_batch(tokenizer,[text])
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    i = int(probs.argmax())
    return {"label":LABELS[i],"confidence":probs[i].item()}