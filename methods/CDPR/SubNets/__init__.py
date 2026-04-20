from .FeatureNets import BERTEncoder

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'bert-large-uncased': BERTEncoder,
                }