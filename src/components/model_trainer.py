import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_trainer_config = (
            model_trainer_config
        )

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifact:

        try:

            logging.info(
                "Loading transformed train and test arrays"
            )

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(
                "Training Random Forest model"
            )

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted"
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted"
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted"
            )

            logging.info(
                f"Accuracy : {accuracy}"
            )

            metric_artifact = (
                ClassificationMetricArtifact(
                    accuracy_score=accuracy,
                    f1_score=f1,
                    precision_score=precision,
                    recall_score=recall
                )
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                model
            )

            logging.info(
                "Model saved successfully"
            )

            model_trainer_artifact = (
                ModelTrainerArtifact(
                    trained_model_file_path=
                    self.model_trainer_config.trained_model_file_path,

                    metric_artifact=
                    metric_artifact
                )
            )

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)