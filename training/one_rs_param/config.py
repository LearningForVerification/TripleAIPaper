import configparser
import os

def load_config(file_name):
    # Percorso del file .ini
    config_path = os.path.join(os.path.dirname(__file__), f'configs/{file_name}')
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config



