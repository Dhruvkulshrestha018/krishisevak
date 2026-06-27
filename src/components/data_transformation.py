import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.exception import MyException
from src.logger import logging

from src.constants import SCHEMA_FILE_PATH

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):

        try:

            self.data_ingestion_artifact = data_ingestion_artifact

            self.data_transformation_config = (
                data_transformation_config
            )

            self.data_validation_artifact = (
                data_validation_artifact
            )

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str):

        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self):

        try:

            numerical_columns = (
                self._schema_config["numerical_columns"]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "StandardScaler",
                        StandardScaler(),
                        numerical_columns
                    )
                ]
            )

            logging.info(
                "Preprocessor object created successfully"
            )

            return preprocessor

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:

        try:

            logging.info(
                "Data Transformation Started"
            )

            if not self.data_validation_artifact.validation_status:

                raise Exception(
                    self.data_validation_artifact.message
                )

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )

            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            target_column = (
                self._schema_config["target_column"]
            )

            X_train = train_df.drop(
                columns=[target_column]
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=[target_column]
            )

            y_test = test_df[target_column]

            logging.info(
                "Train/Test split into features and target"
            )

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(
                y_train
            )

            y_test = label_encoder.transform(
                y_test
            )

            logging.info(
                "Target column encoded"
            )

            preprocessor = (
                self.get_data_transformer_object()
            )

            X_train_scaled = (
                preprocessor.fit_transform(X_train)
            )

            X_test_scaled = (
                preprocessor.transform(X_test)
            )

            logging.info(
                "Feature scaling completed"
            )

            train_arr = np.c_[
                X_train_scaled,
                y_train
            ]

            test_arr = np.c_[
                X_test_scaled,
                y_test
            ]

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info(
                "Transformation objects saved"
            )

            data_transformation_artifact = (
                DataTransformationArtifact(
                    transformed_object_file_path=
                    self.data_transformation_config.transformed_object_file_path,

                    transformed_train_file_path=
                    self.data_transformation_config.transformed_train_file_path,

                    transformed_test_file_path=
                    self.data_transformation_config.transformed_test_file_path
                )
            )

            logging.info(
                f"Data Transformation Artifact : {data_transformation_artifact}"
            )

            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)