import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
import pandas as pd
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            numerical_col=["writing_score","reading_score"]
            categorical_col=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline=Pipeline(
                steps=[
                    ("Impuetr",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )
            category_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OHE",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and Category pipeline is defined")
            preprocessor=ColumnTransformer(
                [
                    ("NumericalPipeline",num_pipeline,numerical_col),
                    ("CategoricalPipeline",category_pipeline,categorical_col)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            self.data_transformation_config=DataTransformationConfig()
            target_feature='math_score'
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read both train and test data")
            input_train_df=train_df.drop(target_feature,axis=1)
            target_train=train_df[target_feature]
            input_test_df=test_df.drop(target_feature,axis=1)
            target_test=test_df[target_feature]
            logging.info("Seperated input and output features for both train and test")
            preprocessing_obj=DataTransformation().get_data_transformation_object()
            input_train_pre=preprocessing_obj.fit_transform(input_train_df)
            input_test_pre=preprocessing_obj.transform(input_test_df)
            logging.info("Both test and train inputs are preprocessed")
            transformed_train=np.c_[input_train_pre,np.array(target_train)]
            transformed_test=np.c_[input_test_pre,np.array(target_test)]
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                transformed_train,
                transformed_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
