import os
import torch
import yaml


# DEVICE TYPE [CPU/GPU]:
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STR)


# Paths:
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


#
CONFIG_FILE_PTH = os.path.join(SCRIPT_DIR, 'config.yaml')
CONFIG:dict = yaml.load(open(CONFIG_FILE_PTH), Loader=yaml.SafeLoader) 


# Helper func: Resolve paths from config relative to SCRIPT_DIR
def resolve_config_path(config_path_value):
    return os.path.abspath(os.path.join(SCRIPT_DIR, config_path_value)) 

# Other paths [resolved paths]:
LOG_PATH = resolve_config_path(CONFIG['log_path'])
ARTIFACTS_PATH = resolve_config_path(CONFIG['artifacts_path'])
RAW_DATA_PATH = resolve_config_path(CONFIG['raw_data_path'])
PROCESSED_DATA_PATH = resolve_config_path(CONFIG['processed_data_path'])

# Ensure: Output Directories exists -> [logs, artifacts]
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ARTIFACTS_PATH), exist_ok=True)
