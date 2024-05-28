import torch 
import torch.nn as nn 


class Embedding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    

# implement base interface for following Builder pattern or abc class 
class ModelBuilding(object):
    def __init__(self, config) -> None:
        pass


    def build(self):
        pass 

    def run(self):
        pass 