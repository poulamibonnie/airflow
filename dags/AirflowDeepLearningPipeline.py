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
    inputOperatorConfig.model_type = 'BERT'
    extraConfig.attention_mask = ''
    extraConfig.text = '1'
    extraConfig.input_ids = 'xx'
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



