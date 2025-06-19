from constants import *
from ETL.pipeline import ETLPipeline
from ml_pipeline.pipeline import MLPipeline


def main():
    if not os.listdir(PROCESSED_DATA_PATH):
        etl_pipeline = ETLPipeline()
        etl_pipeline.invoke()
    else:
        print('Skipping ETL [Processed News Data Present in Directory.]')

    pipeline = MLPipeline()
    pipeline.invoke()



        
if __name__ == "__main__":
    main()