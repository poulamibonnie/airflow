import tensorflow as tf 
from tensorflow.python.keras.layers.recurrent import SimpleRNN
from tensorflow.python.keras.metrics import RecallAtPrecision

# implement base interface for following Builder pattern or abc class 
class ModelEvaluate(object):
    def __init__(self, config) -> None:
        pass


    def build(self):
        pass 
 
    def run(self):
        pass 