from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig

import sys

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(f"Training data saved at: {train_data}")
        print(f"Test data saved at: {test_data}")

    except Exception as e:
        raise CustomException(e, sys)

        
        

