import os
import yaml
from typing import Optional
from .constants import CONFIG_DIR

class YAMLFile:
    """
    """

    def __init__(self, yaml_filepath: Optional[str] = None) -> None:
        if yaml_filepath is None:
            yaml_filepath  = os.path.join(CONFIG_DIR, "config.YAML") 
        
        yaml_file = self.open_yaml_file(yaml_filepath)
        for k, v in yaml_file.items():
            setattr(self, k, v)
    
    def open_yaml_file(self, yaml_filepath: str) -> dict:
        """
        """
        
        # Check if YAML file exists.
        if not os.path.exists(yaml_filepath):
            raise FileNotFoundError(f"Error: {yaml_filepath} does not exist!")
        
        # Try to open file using context manager and raise ValueError if unable to.
        try:
            with open(yaml_filepath, "r") as file:
                yaml_config_file = yaml.safe_load(file)
        except ValueError:
            print(f"Error: {yaml_filepath} cannot be opened!")

        return yaml_config_file