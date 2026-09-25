from torchao.quantization import quantize_,Int8WeightOnlyConfig
import io,torch,time,copy

def quantize_int8(model):
    m = copy.deepcopy(model)
    quantize_(m,Int8WeightOnlyConfig())
    return m.eval()

def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(),buf)
    return buf.tell() / 1e6

def time_inference(model,x):
    with torch.no_grad():
        before_time = time.perf_counter()
        _ = model(x)
        after_time = time.perf_counter()
    return after_time - before_time
    
