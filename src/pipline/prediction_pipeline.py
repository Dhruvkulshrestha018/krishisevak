import sys
from src.entity.config_entity import CropPredictorConfig
from src.entity.s3_estimator import KrishiSevakEstimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class CropData:
    """
    Creates the input dataframe for crop prediction.
    """

    def __init__(
        self,
        N: int,
        P: int,
        K: int,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float
    ):

        try:
            self.N = N
            self.P = P
            self.K = K
            self.temperature = temperature
            self.humidity = humidity
            self.ph = ph
            self.rainfall = rainfall

        except Exception as e:
            raise MyException(e, sys)


    def get_crop_data_as_dict(self):
        """
        Returns input data as dictionary.
        """

        try:

            input_dict = {

                "N": [self.N],
                "P": [self.P],
                "K": [self.K],
                "temperature": [self.temperature],
                "humidity": [self.humidity],
                "ph": [self.ph],
                "rainfall": [self.rainfall]

            }

            logging.info("Crop input dictionary created.")

            return input_dict

        except Exception as e:
            raise MyException(e, sys)


    def get_crop_input_dataframe(self) -> DataFrame:
        """
        Returns dataframe to feed into the model.
        """

        try:

            return DataFrame(self.get_crop_data_as_dict())

        except Exception as e:
            raise MyException(e, sys)


class CropPredictor:
    """
    Loads model from S3 and performs prediction.
    """

    def __init__(
        self,
        prediction_config: CropPredictorConfig = CropPredictorConfig()
    ):

        try:

            self.prediction_config = prediction_config

        except Exception as e:
            raise MyException(e, sys)


    def predict(self, dataframe: DataFrame):

        try:

            logging.info("Loading model from S3...")

            model = KrishiSevakEstimator(

                bucket_name=self.prediction_config.model_bucket_name,

                model_path=self.prediction_config.model_file_path

            )

            prediction = model.predict(dataframe)

            logging.info("Prediction completed successfully.")

            return prediction

        except Exception as e:

            raise MyException(e, sys)