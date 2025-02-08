FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    scipy==1.11.3 \
    sentencepiece==0.1.99 \
    tqdm==4.66.1 \
    psutil==5.9.5

WORKDIR /app

COPY scripts/load_model.py /app/
COPY configs/model_config.json /app/

CMD ["python", "load_model.py"]