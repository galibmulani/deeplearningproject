import sys
from Xray.components.data_ingestion import DataIngestion
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.exceptions import XRayExceptions
from Xray.logger import logging
from Xray.entity.config_entity import DataIngestionConfig

class TrainingPipeline:
    def __init__(self):
       self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self)->DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
        try:
             
             logging.info("Getting the data from S3 bucket")
             data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
             )

             data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

             logging.info("Got the train set and test set from s3")

             logging.info("Exited the start_data_ingestion method of TrainingPipeline class")

             return data_ingestion_artifact;
        except Exception as e:
            raise XRayExceptions(e, sys)
        
if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.start_data_ingestion()