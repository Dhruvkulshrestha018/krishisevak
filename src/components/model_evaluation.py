import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

from src.constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.s3_estimator import KrishiSevakEstimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def get_best_model(self):

        try:

            estimator = KrishiSevakEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path,
            )

            if estimator.is_model_present(
                self.model_eval_config.s3_model_key_path
            ):
                return estimator

            return None

        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self):

        try:

            logging.info("Loading test dataset")

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            X_test = test_df.drop(columns=[TARGET_COLUMN])

            y_test = test_df[TARGET_COLUMN]

            logging.info("Loading newly trained model")

            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            y_pred = trained_model.predict(X_test)
            logging.info("Actual labels:")
            logging.info(y_test.head())

            logging.info("Prediction:")
            logging.info(y_pred[:5])

            logging.info("Types:")
            logging.info(type(y_test.iloc[0]))
            logging.info(type(y_pred[0]))
            trained_model_f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
            )

            logging.info(
                f"New Model F1 Score : {trained_model_f1}"
            )

            best_model = self.get_best_model()

            if best_model is None:

                logging.info("No production model found")

                return True, trained_model_f1, None

            y_pred_old = best_model.predict(X_test)

            old_model_f1 = f1_score(
                y_test,
                y_pred_old,
                average="weighted",
            )

            logging.info(
                f"Production Model F1 Score : {old_model_f1}"
            )

            if trained_model_f1 > old_model_f1:

                return (
                    True,
                    trained_model_f1,
                    old_model_f1,
                )

            return (
                False,
                trained_model_f1,
                old_model_f1,
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:

        try:

            (
                is_model_accepted,
                trained_score,
                old_score,
            ) = self.evaluate_model()

            if old_score is None:

                changed_score = trained_score

            else:

                changed_score = trained_score - old_score

            artifact = ModelEvaluationArtifact(

                is_model_accepted=is_model_accepted,

                changed_accuracy=changed_score,

                s3_model_path=self.model_eval_config.s3_model_key_path,

                trained_model_path=self.model_trainer_artifact.trained_model_file_path,

            )

            logging.info(
                f"Model Evaluation Artifact : {artifact}"
            )

            return artifact

        except Exception as e:
            raise MyException(e, sys)