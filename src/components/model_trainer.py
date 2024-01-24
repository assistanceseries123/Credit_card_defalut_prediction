import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score
    
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from src.exception import CustomException
from src.utils import save_obj,evaluate_model
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,1:-1],
                train_array[:,-1],
                test_array[:,1:-1],
                test_array[:,-1]
            )

            models={
                "Decision tree classifier":DecisionTreeClassifier(),
                "LogisticRegression":LogisticRegression(),
                "RandomforestClassifier":RandomForestClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingclassifier":GradientBoostingClassifier(),
                "KNeigboursClassifier":KNeighborsClassifier(),
                "ExtratreeClassifier":ExtraTreesClassifier()

            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            ##To get best model from the dict
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on training ans the testing dataset")
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            accuracy_score_=accuracy_score(y_test,predicted)
            #f1_=f1_score(y_test,predicted)
            #precision_=precision_score(y_test,predicted)
            #classi_=classification_report(y_test,predicted)
            return accuracy_score_


        except Exception as e:
            raise CustomException(e,sys)