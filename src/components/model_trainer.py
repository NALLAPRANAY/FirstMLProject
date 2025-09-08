import os
import sys
from src.utils import save_obj,evaluate_model
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig,DataTransformation
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_obj_file_path=os.path.join("artifacts","model")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig
    def intiate_model_training(self,train,test):
        try:
            X_train,y_train,X_test,y_test=(
                train[:,:-1],
                train[:,-1],
                test[:,:-1],
                test[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "lasso":Lasso(),
                "SVR":SVR(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "Gradient":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "KNN":KNeighborsRegressor(),
                "XGB":XGBRegressor()
                }
            model_details: dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
           
            best_model_score=max(model_details.values())
            best_model_name=list(model_details.keys())[list(model_details.values()).index(best_model_score)]
            if best_model_score<0.6:
                raise CustomException("Not Modle with better efficiency is found")
            logging.info("Model is trained")

            save_obj(
                file_path=self.model_trainer_config.model_obj_file_path,
                obj=best_model_name
            )

            models=[best_model_name,best_model_score]
            return models

        except Exception as e:
            raise CustomException(e,sys)