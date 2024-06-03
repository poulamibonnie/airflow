from typing import Dict , Any
from dataclasses import dataclass, asdict, field
from typing import Union
import torch 

@dataclass
class ExtraConfig:
    input_ids: object =   None,
    attention_mask : object =  None,
    text: object = None 
    
    def dict(self):
        return {k: v for k, v in asdict(self).items()}

@dataclass
class TextModelConfig(object):
    model_type: str = ""

    # data ingestion
    url: str = ""
    bert_model_name: str = ""
    num_classes: int = 0
    max_length: int = 0
    batch_size: int = 0
    num_epochs: int = 0
    learning_rate: int = 0
    
    # all lstm config params
    input_dim: int = 0
    embedding_dim: int = 0 
    hidden_dim: int = 0
    output_dim: int = 0

    input_size: int = 0 
    hidden_size: int = 0
    num_layers: int = 0
    dropout: float = 0.0
    bidirectional: bool = False
    batch_first: bool = False
    output_size: int = 0 


    pretrained_model_name: str = '' # generic useful as a transformer for both rnn and bert 
    # bert config values
    max_seq_length: int = 0

    # modeule build for both toruch
    loss_function: str = ''
    
    # optimization parameters 
    optimizer_type: str = ''
    optimizer_learn_rate: float = 0.0 
    optimizer_weight_loss: float = 0.0

    # useful for lstm rnn 
    scheduler_type: str = ''
    scheduler_warmup: int = 0

    # evaluate the model 
    input_text_string: str = ''
    save_path: str = ''

    # deploy model
    deployment_type: str = ''
    deployment_conf: dict = field(default_factory=dict)

    device: str = ""
    def dict(self):
        dict_args =  {k: v for k, v in asdict(self).items()}
        dict_args['device'] = deviceInfo()
        return dict_args 

# this is also possible to fecthced from dag config for kwargs passed as task instance
# can be done via dataclass or pydantic class
@dataclass
class OperatorConfig(object):
    task_instance: object = None
    dag_id: object = None 
    task_id: int = 0


    def dict(self):
        return {k: v for k, v in asdict(self).items()}


def deviceInfo():
    import sys 
    if sys.platform == 'darwin': return 'mps' 
    return 'cpu' if not torch.cuda.is_available() else 'gpu'