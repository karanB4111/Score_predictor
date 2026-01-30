from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.mlproject.components.model_trainer import ModelTrainerConfig,Modeltrainer



import sys

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

        model_trainer = Modeltrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))




    except Exception as e:
        raise CustomException(e, sys)

        
        
