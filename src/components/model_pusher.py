import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import KrishiSevakEstimator
from src.exception import MyException
from src.logger import logging


class ModelPusher:

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):

        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.s3 = SimpleStorageService()

        self.estimator = KrishiSevakEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:

        try:

            logging.info("Starting Model Pusher Component")

            if not self.model_evaluation_artifact.is_model_accepted:

                logging.info(
                    "Model was not accepted. Skipping upload."
                )

                return ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=None,
                )

            logging.info("Uploading model to S3")

            self.estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )

            logging.info("Model uploaded successfully")

            artifact = ModelPusherArtifact(

                bucket_name=self.model_pusher_config.bucket_name,

                s3_model_path=self.model_pusher_config.s3_model_key_path,

            )

            logging.info(f"Model Pusher Artifact : {artifact}")

            return artifact

        except Exception as e:
            raise MyException(e, sys) from e