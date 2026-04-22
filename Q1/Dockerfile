FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    torch==2.2.2 \
    sentencepiece==0.2.0 \
    sacrebleu==2.4.2 \
    tqdm

COPY translate.py .
COPY evaluate.py .

CMD ["bash", "-c", "python translate.py && python evaluate.py"]
