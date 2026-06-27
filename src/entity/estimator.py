import sys
import pandas as pd

from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


class MyModel:

    def __init__(
        self,
        preprocessing_object: Pipeline,
        trained_model_object: object
    ):

        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(
        self,
        dataframe: pd.DataFrame
    ):

        try:

            logging.info("Starting prediction")

            transformed_data = (
                self.preprocessing_object.transform(dataframe)
            )

            prediction = (
                self.trained_model_object.predict(
                    transformed_data
                )
            )

            return prediction

        except Exception as e:
            raise MyException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"