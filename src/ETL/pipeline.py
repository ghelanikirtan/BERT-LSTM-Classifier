import os, warnings
import re
from typing import Any
import pandas as pd
from constants import RAW_DATA_PATH, PROCESSED_DATA_PATH
from ETL import MERGE_CATEGORIES_MAPPING, REQUIRED_COLUMS
warnings.filterwarnings('ignore')

class RawDataExtractor:
    def __init__(self, raw_data_file : str = 'news_data.json'):
        self.raw_data_file = raw_data_file

    def extract(self) -> pd.DataFrame:
        return pd.read_json(os.path.join(RAW_DATA_PATH, self.raw_data_file), lines=True)

class TransformationPipe:
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = data
        df['category'] = df['category'].apply(lambda x: MERGE_CATEGORIES_MAPPING.get(x, x))
        df = df[REQUIRED_COLUMS]
        df['processed_news'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
        df['processed_news'] = df['processed_news'].apply(lambda txt: self.remove_noise(txt))
        return df
        
    @staticmethod
    def remove_noise(text) -> str:
        rx = {
            'URL_NOISE_RE' : r"http?:\S+|https?:\S+|www\S+",
            'EMAIL_NOISE_RE' : r"\S+@\S+",
            'HTML_TAGS_NOISE_RE' : r"<.*?>",
            # 'DIG_SC_RE' : r"[^a-z\s]",
            'SC_RE' : r'[^a-zA-Z0-9\s]',
            'WHITESPACE_RE' : r"\s+",
        }
        text = text.lower()
        for noise, expression in rx.items():
            text = re.sub(expression, ' ' if noise == 'WHITESPACE_RE' else '', text)
        text = text.strip()
        return text


class ETLPipeline:

    def __init__(self):
        self.raw_data_extractor = RawDataExtractor() # Extractor 
        self.data_processor = TransformationPipe() # Transformation
        
    def invoke(self, output_file='news_transformed_data.json'):
        
        print('ETL Pipeline: Invoked')
        extracted_data = self.raw_data_extractor.extract()
        print(f'Extracted Raw Data: {extracted_data.shape}')
        transformed_data: pd.DataFrame = self.data_processor.process(extracted_data)
        print(f"Transformation: [text preprocessing, categories combination, headline+short_description]")
        transformed_data.to_json(os.path.join(PROCESSED_DATA_PATH, output_file), orient='records', indent=2)
        print(f"Data Loaded: {os.path.join(PROCESSED_DATA_PATH, output_file)}")
        print(f'ETL Pipeline: Executed Successfully')         

        
# Optional [Text Processing: Transfomation Pipe]:
class TextProcessor:
    def __init__(self):
        
        import nltk, re
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        nltk.download('all')
        
        # Noise Removal Regular Expressions:
        self.rx = {
            'URL_NOISE_RE' : r"http?:\S+|https?:\S+|www\S+",
            'EMAIL_NOISE_RE' : r"\S+@\S+",
            'HTML_TAGS_NOISE_RE' : r"<.*?>",
            'DIG_SC_RE' : r"[^a-z\s]",
            'WHITESPACE_RE' : r"\s+",
        }

        # Stop words: 
        self.stop_words = set(stopwords.words('english'))
        
        # Stemmers & Lemmatizers:
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def process(self, 
                text, 
                stemming=False, 
                lemmatizing=True,
                return_tokens=False
                ) -> Any:
        
        from nltk.tokenize import word_tokenize

        text = text.lower()
        for noise, expression in self.rx.items():
            text = re.sub(expression, ' ' if noise == 'WHITESPACE_RE' else '', text)
        text = text.strip()
        
        tokens = word_tokenize()
        tokens = [word for word in tokens if word not in self.stop_words]
        
        if stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        if lemmatizing:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens if return_tokens else ' '.join(tokens) 
        
        