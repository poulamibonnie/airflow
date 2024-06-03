FROM apache/airflow:slim-latest-python3.9 as airflow

USER root

RUN apt-get update && apt-get install -y \
    gcc clang  \
    pkg-config \
    libhdf5-dev git wget cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

USER airflow

RUN pip install --upgrade pip &&  \
    pip install torch torchvision requests

# install addon nlp text dependencies
RUN pip install "cython<3.0.0" wheel  && pip install "pyyaml==5.4.1" --no-build-isolation
RUN pip install torchtext 

# install any addon dependencies apart from core dl and airflow providers
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
