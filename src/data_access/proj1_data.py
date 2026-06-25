import sys
import pandas as pd
import numpy as np
from typing import Optional
import logging
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class MongoDBDataExporter:

    def __init__(self):
        self.mongo_client = MongoDBClient(
            database_name=DATABASE_NAME
        )

    def export_collection_as_dataframe(
        self,
        collection_name: str
    ) -> pd.DataFrame:

        collection = self.mongo_client.database[
            collection_name
        ]

        logging.info(
            f"Fetching data from {collection_name}"
        )

        df = pd.DataFrame(
            list(collection.find())
        )

        if "_id" in df.columns:
            df.drop(
                columns=["_id"],
                inplace=True
            )

        df.replace(
            {"na": np.nan},
            inplace=True
        )

        logging.info(
            f"Fetched {len(df)} records"
        )

        return df