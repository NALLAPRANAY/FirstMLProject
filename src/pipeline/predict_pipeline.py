import sys
import os
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_path = os.path.join(project_root, 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(project_root, 'artifacts', 'preprocessor.pkl')
            model=load_object(file_path=model_path)
            print(model)
            preprocessor=load_object(file_path=preprocessor_path)
            print(preprocessor)
            data_scaled=preprocessor.transform(features)
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.ethnicity = ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
