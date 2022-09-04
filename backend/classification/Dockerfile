FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

ENV PATH /root/.local/bin:${PATH}

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app/
