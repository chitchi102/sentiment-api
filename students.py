import torch,time
import train,data,tokens,model,quantize,artifacts

def run_student(mode,teacher_path="teacher_logits.pt",n_label=2000,epochs=3,seed=0,out_dir=None,device="cpu"):
    if mode not in ["small","teacher","full"]:
        raise ValueError(f"{mode} must be in small,teacher or full!")
    teacher_logits = torch.load(teacher_path)["train_logits"]
    data_size = len(teacher_logits)
    x_all_train,y_all_train,x_validation,y_validation = data.load_data(n_train=data_size)
    if len(x_all_train) != data_size:
        raise ValueError("not match datasize")
    tokenizer = tokens.build_tokenizer(x_all_train)
    x_train = tokens.encode_batch(tokenizer,x_all_train)
    x_val = tokens.encode_batch(tokenizer,x_validation)
    y_train = torch.tensor(y_all_train)
    y_val = torch.tensor(y_validation)
    torch.manual_seed(seed)
    m = model.SentimentClassifier(tokenizer.get_vocab_size())
    before_time = time.perf_counter()
    if mode == "small":
        _,history = train.train_model(m,x_train[0:n_label],y_train[0:n_label],epochs=epochs,device=device)
    elif mode == "teacher":
        _,history = train.train_distill(m,x_train,teacher_logits,epochs=epochs,device=device)
    else :
        _,history = train.train_model(m,x_train,y_train,epochs=epochs,device=device)    
    after_time = time.perf_counter()
    acc = train.evaluate(m,x_val,y_val,device=device)
    train_seconds = after_time-before_time
    m.to(device="cpu")
    quantized_m = quantize.quantize_int8(m)
    int8_acc = train.evaluate(quantized_m,x_val,y_val,device="cpu")
    if out_dir is not None:
        artifacts.save_artifacts(m,tokenizer,out_dir)
    return {"mode":mode,"n_texts":data_size,"acc":acc,"int8_acc":int8_acc,"train_seconds":train_seconds,"history":history,"device":device}





