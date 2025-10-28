# config data, artifact
# load data
# transform separate train and test or prepare and time to time log
# initiate data transformation
from Xray.entity.artifact_entity import (DataIngestionArtifact,DataTransformationArtifact)
from Xray.entity.config_entity import DataTransformationConfig
from torchvision import transforms
from Xray.exceptions import XRayExceptions
import sys
from Xray.logger import logging
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import os
import joblib
from torchvision.datasets import ImageFolder

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_ingestion_artifact:DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
    
    def transforming_training_data(self)->transforms.Compose:
        try:
            logging.info("Started the transforming_training_data method of DataTransformation class")
            train_transforms:transforms.Compose = transforms.Compose([
                transforms.Resize(self.data_transformation_config.RESIZE),
                transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                transforms.ColorJitter(
                    **self.data_transformation_config.color_jitter_transforms
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(
                    self.data_transformation_config.RANDOMROTATION
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    **self.data_transformation_config.normalize_transforms
                ),
              ]
            )
            logging.info("Completed the transforming_training_data method of DataTransformation class")
            
            return train_transforms;
        except Exception as e:
            raise XRayExceptions(e, sys)
    
    def transforming_testing_data(self)->transforms.Compose:
        try:
            logging.info("Started the transforming_testing_data method of DataTransformation class")    
            test_transforms:transforms.Compose = transforms.Compose([
                transforms.Resize(self.data_transformation_config.RESIZE),
                transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                transforms.ToTensor(),
                transforms.Normalize(
                    **self.data_transformation_config.normalize_transforms
                ),
              ]
            )
            
            logging.info("Completed the transforming_testing_data method of DataTransformation class")

            return test_transforms;
        except Exception as e:
            raise XRayExceptions(e, sys)
        
    def data_loader(self,train_transform:transforms.Compose,test_transform:transforms.Compose)->Tuple[DataLoader,DataLoader]:
        try:
            logging.info("Started the data_loader method of DataTransformation class")

            #train data loader
            train_data:Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform = train_transform,
            )
            
            #test data loader
            test_data:Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform = test_transform,
            )
            
            logging.info("created the train data and test data path")

            train_loader:DataLoader = DataLoader(
                dataset = train_data,
                **self.data_transformation_config.data_loader_params
            )
            test_loader:DataLoader = DataLoader(
                dataset= test_data,
                **self.data_transformation_config.data_loader_params
            )
            logging.info("Completed the data_loader method of DataTransformation class")

            return train_loader,test_loader;
    
        except Exception as e:
            raise XRayExceptions(e, sys)
        
    def initiate_data_transformation(self)->None:
        try:
            logging.info("Entered the initiate_data_transformation method of DataTransformation class")

            train_transform:transforms.Compose = self.transforming_training_data()
            test_transform:transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir,exist_ok=True)

            joblib.dump(
                train_transform,
                self.data_transformation_config.train_transforms_file
            ) 
            joblib.dump(
                test_transform,
                self.data_transformation_config.test_transforms_file
            )

            train_loader,test_loader = self.data_loader(train_transform=train_transform,test_transform=test_transform)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_object = train_loader,
                transformed_test_object = test_loader,
                train_transform_file_path = self.data_transformation_config.train_transforms_file,
                test_transform_file_path = self.data_transformation_config.test_transforms_file,
            )
            return data_transformation_artifact;
    
        except Exception as e:
            raise XRayExceptions(e, sys)