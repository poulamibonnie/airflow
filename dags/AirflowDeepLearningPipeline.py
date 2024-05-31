from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from AirflowDeepLearningOperator import AirflowDeepLearningOperator
from OperatorConfig import TextModelConfig, ExtraConfig, OperatorConfig

from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
from typing import Dict , Any


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

    inputOperatorConfig = inputOperatorConfig.dict()
    '''
        TODO: 
            will populate all fields for input in python code raw 
            read it from config file (Later once first integration run is done )
    '''

    ti.xcom_push(key='model_input', value=inputOperatorConfig)

with DAG(
    dag_id="Airflow_Deep_Learning_Operator",
    description="test docker run",
    default_args=default_args,
    schedule_interval=None 
) as dag:
    base = BashOperator(task_id="init_run",
                        bash_command="echo 'Starting training the model'")


    operatorInput = PythonOperator(task_id="pass_commands", python_callable=push_args)
    torch_v = AirflowDeepLearningOperator(task_id="version", operatorConfig={
        "dag_id": dag,
        "task_id": 'version'
    })

    base >> operatorInput >> torch_v



