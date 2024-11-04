
from modeling.architectures.MobileNet import MobileNetInitializer
from modeling.architectures.SqueezeNet import SqueezeNetInitializer
from modeling.ModelInitializer import BaseModelInitializer
from modeling.architectures.ShuffleNet import ShuffleNetInitializer
from modeling.architectures.GhostNet import GhostNetInitializer

class ModelInitializerFactory:
    def __call__(self, config:dict) -> BaseModelInitializer:
        """Returns a ModelInitializer depending on the value of config['model_initializer']

        Args:
            config (dict): Must contain 'model_initializer' key. Supported values: 'squeezenet', 'mobilenet'

        Returns:
            BaseModelInitializer: Allows CNN class to initialize different architectures
        """
        if 'model_initializer' not in config.keys():
            raise KeyError('model_initializer not found in config')
        
        match config['model_initializer']:
            case "squeezenet":
                return SqueezeNetInitializer(config)
            case "mobilenet":
                return MobileNetInitializer(config)
            case "shufflenet":
                return ShuffleNetInitializer(config)
            case "ghostnet":
                return GhostNetInitializer(config)
            case _:
                raise ValueError(f"model type {config['model_initializer']} does not exist!")