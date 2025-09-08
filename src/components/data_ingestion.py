import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def intiate_data_ingestion(self):
        logging.info("Data ingestion is started")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Data ingestion is completed")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("Train test split is intiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Train test split is completed and data is stored in artifacts folder")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.intiate_data_ingestion()

    data_transformation=DataTransformation()
    train,test,_=data_transformation.intiate_data_transformation(train_path,test_path)

    modeltrainer=ModelTrainer()
    modeldetails=modeltrainer.intiate_model_training(train,test)
    print(modeldetails)
