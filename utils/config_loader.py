import copy
import yaml
'''
demo:
====================
layers:
  - in_features: 3
    out_features: 16
    num_heads: 2
    activation: "relu"
  - in_features: 32
    out_features: 32
    num_heads: 1
    activation: "relu"
  - in_features: 32
    out_features: 1
    num_heads: 1
    activation: "sigmoid"
====================
'''


class LayerConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            raw_config = yaml.safe_load(file)

        config = {"layers": []}
        layers = raw_config["layers"]

        for i, layer in enumerate(layers):
            repeat = layer.get("repeat", 1)
            for _ in range(repeat):
                config["layers"].append(copy.deepcopy(layer))

        config["global_batch_norm"] = raw_config.get("global_batch_norm", False)

        return config

    def get_in_features(self):
        return self.config["layers"][0]["args"]["in_features"]
    
    def get_out_features(self):
        return self.config["layers"][-1]["args"]["out_features"]

    def update_config(self, key: str, value):
        self.config[key] = value

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


class LossConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            raw_config = yaml.safe_load(file)

        return raw_config["losses"]

    def update_config(self, key: str, value):
        self.config[key] = value

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
