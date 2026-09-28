import torch,json,time,data,teacher,quantize,artifacts,tokens,bench

def read_jsonl(path):
    dict_list = []
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dict_list.append(json.loads(line))
    return dict_list

def acc_from_logits(logits,y):
    return ((logits.argmax(dim=1) == y) / len(y)).sum().item()

def summarize(results,mode,key="acc"):
    acc_list = []
    for r in results:
        if r["mode"] == mode:
            acc_list.append(r[key])
    if len(acc_list) == 0:
        raise ValueError("dictionary is not exist")
    return {"acc":sum(acc_list)/len(acc_list),"acc_min":min(acc_list),"acc_max":max(acc_list),"n_seeds":len(acc_list)}

def teacher_row(val_logits,y_val,size_mb,seconds):
    acc = acc_from_logits(val_logits,y_val)
    return {"name":"teacherBERT","acc":acc,"acc_min":acc,"acc_max":acc,"n_seeds":1,"size_mb":size_mb,"seconds":seconds}

def student_rows(results,bench_rows):
    names = ["small","teacher","full"]
    result = []
    for name in names:
        s = summarize(results,name)
        acc,acc_min,acc_max,n_seeds = s["acc"],s["acc_min"],s["acc_max"],s["n_seeds"]
        size_mb,seconds = bench_rows[0]["size_mb"],bench_rows[0]["seconds"]
        result.append({"name":name,"acc":acc,"acc_min":acc_min,"acc_max":acc_max,"n_seeds":n_seeds,"size_mb":size_mb,"seconds":seconds})
    s = summarize(results,"teacher",key="int8_acc")
    acc,acc_min,acc_max,n_seeds = s["acc"],s["acc_min"],s["acc_max"],s["n_seeds"]
    size_mb,seconds = bench_rows[1]["size_mb"],bench_rows[1]["seconds"]
    result.append({"name":"teacher_int8","acc":acc,"acc_min":acc_min,"acc_max":acc_max,"n_seeds":n_seeds,"size_mb":size_mb,"seconds":seconds})
    return result

def to_markdown(rows):
    result = ["| モデル | val 正解率 | サイズ (MB) | 推論 (秒) |","|---|---|---|---|"]
    for r in rows:
        if r['n_seeds'] > 1:
            result.append(f"| {r['name']} | {r['acc']:.3f} ({r['acc_min']:.3f}-{r['acc_max']:.3f}) | {r['size_mb']:.1f} | {r['seconds']:.2f} |")
        else:
            result.append(f"| {r['name']} | {r['acc']:.3f} | {r['size_mb']:.1f} | {r['seconds']:.2f} |")
    return "\n".join(result)

if __name__ == "__main__":
    results = read_jsonl("results/cycle3_s3_colab_t4.jsonl")
    _,_,x_validation,y_validation = data.load_data(n_train=1)
    y_val = torch.tensor(y_validation)
    t_tokenizer, t_model = teacher.load_teacher()
    before_time = time.perf_counter()
    predict = teacher.teacher_logits(t_tokenizer,t_model,x_validation)
    after_time = time.perf_counter()
    val_logits = torch.load("teacher_logits.pt")["val_logits"]
    size_mb = quantize.model_size_mb(t_model)
    teacher_dict = teacher_row(val_logits,y_val,size_mb,after_time-before_time)
    s_model,s_tokenizer = artifacts.load_artifacts("student_teacher")
    vec_x = tokens.encode_batch(s_tokenizer,x_validation)
    b = bench.compare(s_model,vec_x,y_val)
    students_dict = student_rows(results,b)
    chart = to_markdown([teacher_dict] + students_dict)
    with open("results/cycle3_table.md","w",encoding="utf-8") as f:
        f.write(chart + "\n")
    print(chart)