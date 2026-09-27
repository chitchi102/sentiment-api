import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import data

TEACHER_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

def load_teacher(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME)
    model.eval().to(device)
    return tokenizer,model


def teacher_logits(tok,m,texts,batch_size=64,device="cpu"):
    with torch.no_grad():
        predict = []
        for i in range(0,len(texts),batch_size):
            x = tok(texts[i:i+batch_size],padding=True,truncation=True,max_length=64,return_tensors="pt")
            x = x.to(device)
            predict.append(m(**x).logits.cpu())
        predict = torch.cat(predict)
    return predict

def save_teacher_logits(path,n_train,device="cpu",batch_size=64):
    X_train,_,X_validation,_ = data.load_data(n_train=n_train)
    tokenizer,model = load_teacher(device)
    train_predict = teacher_logits(tokenizer,model,X_train,batch_size,device)
    validation_predict = teacher_logits(tokenizer,model,X_validation,batch_size,device)
    torch.save({"train_logits":train_predict,
                "val_logits":validation_predict},path)

