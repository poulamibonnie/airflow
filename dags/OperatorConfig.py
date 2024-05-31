from typing import Dict , Any
from dataclasses import dataclass 
from typing import Union


@dataclass
class TextModelConfig(dataclass):
    model_type: str 

    # data ingestion
    url: str
    bert_model_name: str
    num_classes: int
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: int
    
    # all lstm config params
    input_dim: int 
    embedding_dim: int 
    hidden_dim: int 
    output_dim: int 

    input_size: int 
    hidden_size: int 
    num_layers: int 
    dropout: float
    bidirectional: bool
    batch_first: bool
    output_size: int 


    pretrained_model_name: str # generic useful as a transformer for both rnn and bert 
    # bert config values
    num_classes: int 
    max_seq_length: int 

    # modeule build for both toruch
    batch_size: int 
    num_epochs: int 
    loss_function: str 
    
    # optimization parameters 
    optimizer_type: str 
    optimizer_learn_rate: float 
    optimizer_weight_loss: float

    # useful for lstm rnn 
    scheduler_type: str 
    scheduler_warmup: int 


    # evaluate the model 
    input_text_string: str 
    save_path: str 


@dataclass
class OperatorConfig(object):
    task_id: int 
    task_instance: Any
    dag_id: Any