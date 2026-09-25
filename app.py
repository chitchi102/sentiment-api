from fastapi import FastAPI
from pydantic import BaseModel
import predict

import artifacts,quantize

model, tokenizer = artifacts.load_artifacts()
model = quantize.quantize_int8(model)

app = FastAPI()
class PredictIn(BaseModel):
    text:str

@app.get("/healthz")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict_endpoint(body: PredictIn):
    return predict.predict(model,tokenizer,body.text)
