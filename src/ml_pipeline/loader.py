import os
import pandas as pd
from constants import PROCESSED_DATA_PATH



# Load Data:
def load_data(filepath = PROCESSED_DATA_PATH, filename:str = 'news_transformed_data.json') -> pd.DataFrame:
    return pd.read_json(os.path.join(filepath, filename))
    



