FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py artifacts.py predict.py tokens.py model.py quantize.py model.pt tokenizer.json ./

EXPOSE 7860

CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]