from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from AirflowDeepLearningOperator import AirflowDeepLearningOperator
from OperatorConfig import TextModelConfig, ExtraConfig, OperatorConfig

from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
from typing import Dict , Any

import json, os 

default_args = {
    'owner': 'synarcs',
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'start_date': days_ago(1)
}

def push_args(**kwargs):
    ti = kwargs['ti']
    print(kwargs)
    inputOperatorConfig: TextModelConfig = TextModelConfig()
    extraConfig: ExtraConfig = ExtraConfig()
    
    # data ingestion config 
    inputOperatorConfig.url = 'https://storage.googleapis.com/kaggle-data-sets/134715/320111/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240601%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240601T200759Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=38a2c000f592607c02707838ed88d3519bc2e39267d17a8216d011ebb1374b9bb94fc334562f21b3ee448659db07888651913f11532bd38666a45a15143997fcdf73eb4642bd8f11336fbd726a00e42c76288b370ebade5dbb6989b7de691fee0db7d19de963f2d318c9320ea47874a96e3ba59eef8e383df49d6fd95455ac02889dac14eb8f4698110f941262c5124e0cf4a4b6052bfcfe8d8bf6edfeb36b906d292dd5616d504d316e4b9bb9fb6dfc4420723189fd9cd893470a858d30cc1bcab068f2dc20f9b17db9c8c2db833f204d33a89f923b767ad0f501893d86fbabc0d063c88e767b72ad259196f56a9c5b181efa5591f0d0327f68576d6d4c65b3'
    inputOperatorConfig.model_type = 'BERT'
    inputOperatorConfig.num_classes = 2
    inputOperatorConfig.max_length = 128 
    inputOperatorConfig.batch_size = 16
    inputOperatorConfig.num_epochs = 1
    inputOperatorConfig.learning_rate = 2e-5

    # model building config 
    inputOperatorConfig.bert_model_name = 'distilbert-base-uncased'
    inputOperatorConfig.dropout = 0.1

    extraConfig.attention_mask = ''
    extraConfig.text = ''
    extraConfig.input_ids = ''
    extraConfig = extraConfig.dict()

    inputOperatorConfig.extra_config = extraConfig 
    inputOperatorConfig = inputOperatorConfig.dict()
    '''
        TODO: 
            will populate all fields for input in python code raw 
            read it from config file (Later once first integration run is done )
    '''

    ti.xcom_push(key='model_input', value=inputOperatorConfig)

def getOperatorDagConfig(dag: DAG) -> Dict[str, object]:
    return {
        "dag_id": dag,
        "task_id": 'version'
    }

with DAG(
    dag_id="Airflow_Deep_Learning_Operator",
    description="test docker run",
    default_args=default_args,
    schedule_interval=None 
) as dag:
    base = BashOperator(task_id="init_run",
                        bash_command="echo 'Starting training the model'")


    operatorInput = PythonOperator(task_id="pass_commands", python_callable=push_args)
    baseOperatorConfig: Dict[str, object] = getOperatorDagConfig(dag=dag) 
    torch_v = AirflowDeepLearningOperator(task_id="torch_dll",operatorConfig=baseOperatorConfig)

    base >> operatorInput >> torch_v



