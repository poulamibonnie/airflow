#!/bin/bash 

echo "[x] Building the custom docker image consisting of Pytorch" 
docker build -t airflow-torch:2.9.1 . 
echo "[x] Starting Airflow Services" 
docker-compose up  


