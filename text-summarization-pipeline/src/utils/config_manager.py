import yaml
import os

class ConfigManager:
    """Handles loading and management of configuration files."""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing config file: {str(e)}")
            
    def get_config(self):
        """Return the loaded configuration."""
        return self.config
        
    def validate_config(self):
        """Basic configuration validation."""
        required_keys = ['dataset', 'model', 'preprocessing']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config section: {key}")
