import torch,os
import quantize,train,artifacts
import data,tokens


def compare(model,x,y):
    quantized_model = quantize.quantize_int8(model)
    return [{"name":"fp32","size_mb":quantize.model_size_mb(model),"acc":train.evaluate(model,x,y),"seconds":quantize.time_inference(model,x)},
            {"name":"int8","size_mb":quantize.model_size_mb(quantized_model),"acc":train.evaluate(quantized_model,x,y),"seconds":quantize.time_inference(quantized_model,x)}]

if __name__ == "__main__":
    out_dir = "."
    model,tokenizer = artifacts.load_artifacts(out_dir)
    x_train,y_train,x_validation,y_validation = data.load_data(n_train=1)
    x_validation = tokens.encode_batch(tokenizer,x_validation)
    fp32 , int8 = compare(model,x_validation,torch.tensor(y_validation))
    print(fp32,int8)