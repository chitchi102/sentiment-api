import torch
import torch.nn as nn
import time 

import data ,tokens ,model ,artifacts


def train_model(model,x,y,epochs=3,batch_size=64,lr=3e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    history = []
    n = len(y)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0,n,batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[idx]),y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total/n)
    return model,history

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        predict = model(x).argmax(dim=1)
        accuracy = (predict==y).sum() / len(y)
    return accuracy.item()

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
    
    