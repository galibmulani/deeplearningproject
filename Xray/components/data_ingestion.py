import sys

from Xray.cloud_storage.s3_operation import S3Operation
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig
from Xray.exceptions import XRayExceptions
from Xray.logger import logging


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()

    def get_data_from_s3(self)->None:
        try:
            self.s3.sync_folder_from_s3(
                folder = self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                folder_bucket_name= self.data_ingestion_config.S3_data_folder
            )
        except Exception as e:
            raise XRayExceptions(e, sys)
    
    
    def initiate_data_ingestion(self):
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")
        try:
            self.get_data_from_s3()

            dataingestion_artifact:DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_path,
                test_file_path=self.data_ingestion_config.testing_path
            )

            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")
            return dataingestion_artifact;  
    
        except Exception as e:
            raise XRayExceptions(e, sys)