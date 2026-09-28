import torch
import torch.nn as nn
import time 

import data ,tokens ,model ,artifacts,distill


def train_model(model,x,y,epochs=3,batch_size=64,lr=3e-4,device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    history = []
    n = len(y)
    model.train().to(device)
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0,n,batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            x_=x[idx].to(device)
            y_=y[idx].to(device)
            loss = criterion(model(x_),y_)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total/n)
    return model,history

def evaluate(model,x,y,device="cpu"):
    model.eval().to(device)
    with torch.no_grad():
        x_=x.to(device)
        y_=y.to(device)
        predict = model(x_).argmax(dim=1)
        accuracy = (predict==y_).sum() / len(y_)
    return accuracy.item()

def train_distill(model,x,teacher_logits,epochs=3,batch_size=64,lr=3e-4,T=2.0,device="cpu"):
    model.train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    history = []
    n = len(x)
    for _ in range(epochs):
        total = 0.0
        perm = torch.randperm(n)
        for i in range(0,n,batch_size):
            idx = perm[i:i+batch_size]
            x_=x[idx].to(device)
            t_=teacher_logits[idx].to(device)
            optimizer.zero_grad()
            loss = distill.distill_loss(model(x_),t_,T)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            
        history.append(total/n)
    return model,history



def run(n_train=20000,epochs=3,out_dir=None):
    X_train,y_train,X_validation,y_validation = data.load_data(n_train=n_train)
    tokenizer = tokens.build_tokenizer(X_train)
    x_train = tokens.encode_batch(tokenizer=tokenizer,texts=X_train)
    x_validation = tokens.encode_batch(tokenizer=tokenizer,texts=X_validation)
    mod = model.SentimentClassifier(tokenizer.get_vocab_size())
    before_accuracy = evaluate(mod,x_validation,torch.tensor(y_validation))
    before_time = time.time()
    _,history = train_model(mod,x_train,torch.tensor(y_train),epochs=epochs)
    after_time = time.time()
    after_accuracy = evaluate(mod,x_validation,torch.tensor(y_validation))
    params = model.count_params(mod)
    train_seconds = after_time-before_time
    if out_dir is not None:
        artifacts.save_artifacts(mod,tokenizer,out_dir)
    return {"acc_before":before_accuracy,
            "acc_after":after_accuracy,
            "params":params,
            "train_seconds":train_seconds,
            "history":history}

if __name__ == "__main__":
    result = run(out_dir=".")
    print(result)
    
    