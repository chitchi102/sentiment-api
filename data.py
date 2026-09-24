from datasets import load_dataset

def load_data(n_train=20000):
    ds = load_dataset("stanfordnlp/sst2")
    X_train = ds["train"]["sentence"][:n_train]
    y_train = ds["train"]["label"][:n_train]
    X_validation = list(ds["validation"]["sentence"])
    y_validation = list(ds["validation"]["label"])
    return (X_train,y_train,X_validation,y_validation)