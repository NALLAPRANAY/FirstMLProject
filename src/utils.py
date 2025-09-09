import os
import sys
import pickle
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(X_train,y_train,X_test,y_test,models):
    model_values={}
    for model in range(len(list(models.keys()))):
        train_model=list(models.values())[model]
        train_model.fit(X_train,y_train)
        y_pred=train_model.predict(X_test)
        score=r2_score(y_test,y_pred)
        model_values[list(models.keys())[model]]=score
    return model_values
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

