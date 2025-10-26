import os
from dataclasses import dataclass
from torch import device
from Xray.constant.training_pipeline import *



@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.S3_data_folder: str = S3_DATA_FOLDER
        self.bucket_name: str = BUCKET_NAME
        self.artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
        self.data_path: str = os.path.join(self.artifact_dir, "data_ingestion",self.S3_data_folder)
        self.training_path: str = os.path.join(self.data_path,"train")
        self.testing_path: str = os.path.join(self.data_path,"test")