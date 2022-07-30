FROM tiangolo/uvicorn-gunicorn:python3.8-slim

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app/

