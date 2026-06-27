# train_set, test_set = train_test_split( dataframe,test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)


from src.pipline.training_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()