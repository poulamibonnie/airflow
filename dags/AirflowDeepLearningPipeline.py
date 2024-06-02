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
    inputOperatorConfig.url = 'https://storage.googleapis.com/kaggle-data-sets/134715/320111/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240602%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240602T062033Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4b229c3fcb63cce7ef415547154192dd0a6f4c04de5b0a8e2db324b3c5f465ab01d7589c084784be1ebdcac28b97ce50c560d89b3307c393742a45e75c5b956d754afc131339e908e4c007b2665b984be90a2e787b3df8b9177cc5bb226d4dbff26a9378c4533097e45941ee73c997d362e16fc7e6745c9a682eb55da339bf9d65314a04d0e028688fbcf67c1f91eb68b24b00d758f011a98c22137d18902bec121a2564dae892d18749b80562a69db7b2ca77411666e23924c83735ea165e11626ae7ba2c20a4eb6988988fd5e0ac69169bc1ed3a406da8b052af22ce3c558321604f2eea661cac0fa2f25b66b1674f9ed36e53371fbd13a025993f6107d424'
    inputOperatorConfig.model_type = 'BERT'
    inputOperatorConfig.num_classes = 2
    inputOperatorConfig.max_length = 128 
    inputOperatorConfig.batch_size = 16
    inputOperatorConfig.num_epochs = 1
    inputOperatorConfig.learning_rate = 2e-5

    # model building config 
    inputOperatorConfig.bert_model_name = 'distilbert-base-uncased'
    inputOperatorConfig.dropout = 0.1

    inputOperatorConfig.deployment_type = 'local'
    inputOperatorConfig.deployment_conf = {
        'location': 'model.pkl',
    }

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



