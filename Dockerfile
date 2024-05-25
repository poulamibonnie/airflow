FROM apache/airflow:2.9.1


USER root

RUN apt-get update && apt-get install -y \
    gcc clang  \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

USER airflow

RUN pip install --upgrade pip && \
    pip install --no-cache-dir apache-airflow-providers-postgres && \
    pip install tensorflow requests h5py && \
    pip install tensorflow-datasets
    




