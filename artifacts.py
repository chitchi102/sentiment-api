import os
import torch
from tokenizers import Tokenizer
from model import SentimentClassifier

def save_artifacts(model,tokenizer,out_dir="."):
    torch.save(model.state_dict(),os.path.join(out_dir,"model.pt"))
    tokenizer.save(os.path.join(out_dir,"tokenizer.json"))

def load_artifacts(out_dir="."):
    tokenizer = Tokenizer.from_file(os.path.join(out_dir,"tokenizer.json"))
    model = SentimentClassifier(vocab_size=tokenizer.get_vocab_size())

    model.load_state_dict(torch.load(os.path.join(out_dir,"model.pt")))
    model.eval()
    return model,tokenizer