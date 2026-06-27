import os
import sys
import json
import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def validate_required_columns(
        self,
        dataframe: DataFrame
    ) -> bool:

        required_columns = list(
            self.schema_config["columns"].keys()
        )

        missing_columns = []

        for column in required_columns:
            if column not in dataframe.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            logging.info(
                f"Missing columns: {missing_columns}"
            )
            return False

        return True

    def validate_empty_dataframe(
        self,
        dataframe: DataFrame
    ) -> bool:

        if dataframe.shape[0] == 0:
            return False

        return True

    def initiate_data_validation(
        self
    ) -> DataValidationArtifact:

        try:

            validation_error_msg = ""

            logging.info(
                "Starting Data Validation"
            )

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )

            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            # Check empty dataframe

            if not self.validate_empty_dataframe(train_df):
                validation_error_msg += (
                    "Train dataframe is empty. "
                )

            if not self.validate_empty_dataframe(test_df):
                validation_error_msg += (
                    "Test dataframe is empty. "
                )

            # Check required columns

            if not self.validate_required_columns(
                train_df
            ):
                validation_error_msg += (
                    "Missing columns in train dataframe. "
                )

            if not self.validate_required_columns(
                test_df
            ):
                validation_error_msg += (
                    "Missing columns in test dataframe. "
                )

            validation_status = (
                len(validation_error_msg) == 0
            )

            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.validation_report_file_path
                ),
                exist_ok=True
            )

            report = {
                "validation_status": validation_status,
                "message": validation_error_msg
            }

            with open(
                self.data_validation_config.validation_report_file_path,
                "w"
            ) as file:
                json.dump(
                    report,
                    file,
                    indent=4
                )

            data_validation_artifact = (
                DataValidationArtifact(
                    validation_status=validation_status,
                    message=validation_error_msg,
                    validation_report_file_path=
                    self.data_validation_config.validation_report_file_path
                )
            )

            logging.info(
                f"Validation Artifact : {data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys)