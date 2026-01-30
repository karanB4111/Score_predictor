from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation


import sys

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        # print(f"Training data saved at: {train_data}")
        # print(f"Test data saved at: {test_data}")
        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data, test_data)

    except Exception as e:
        raise CustomException(e, sys)

        
        
