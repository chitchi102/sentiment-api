import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification


TEACHER_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

def load_teacher():
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME)
    model.eval()
    return tokenizer,model


def teacher_logits(tok,m,texts,batch_size=64):
    with torch.no_grad():
        predict = []
        for i in range(0,len(texts),batch_size):
            x = tok(texts[i:i+batch_size],padding=True,truncation=True,max_length=64,return_tensors="pt")
            predict.append(m(**x).logits)
        predict = torch.cat(predict)
    return predict
