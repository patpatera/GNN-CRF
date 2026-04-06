import argparse
import yaml
import os.path as osp
from types import SimpleNamespace


class ConfigLoader:

    DEFAULT_CONFIG_PATH = "./configs/arxiv1.yaml"

    def __init__(self):
        self.config_path = self.__get_config_file()
        self.args = self.__parse_args(self.config_path)

    def __get_config_file(self):
        parser = argparse.ArgumentParser(description="config")
        parser.add_argument("--config", type=str, default="", help="Path to YAML config file with the parameters.")
        parser = parser.parse_args()

        return parser.config

    def __parse_args(self, path):
        config_path = self.__get_config_file()


        args_def = yaml.load(open(ConfigLoader.DEFAULT_CONFIG_PATH), Loader=yaml.SafeLoader)
        
        if osp.exists(config_path):
            args_custom = yaml.load(open(config_path), Loader=yaml.SafeLoader)
            args_def.update(args_custom)
        else:
            print("The config file not found! [" + config_path + "]\n\tUsing default parameters...")

        #args = SimpleNamespace(**args_def)
        print(args_def["experiment"]["name"])
        return args_def
