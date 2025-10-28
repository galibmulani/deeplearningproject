# load data
# model creation train model per epoch
# initiate model training

from Xray.ml.model.arch import Net
from Xray.logger import logging
from Xray.entity.config_entity import ModelTrainerConfig
from Xray.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from Xray.exceptions import XRayExceptions
import sys
from Xray.constant.training_pipeline import *
import joblib
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR,_LRScheduler
import os
import bentoml

class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        self.model_trainer_config : ModelTrainerConfig = model_trainer_config
        self.data_transformation_artifact: DataTransformationArtifact =(
            data_transformation_artifact
        )
        self.model: Module = Net()

    def train(self,optimizer:Optimizer)->None:
        """
        Description: This method is used to train the model
        inputs: model,device,train_loder,optimizer,epochs
        outputs: loss, batch id,accuaracy
        """
        logging.info("Entered the train method of ModelTrainer class")
        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)
            correct: int = 0
            processed = 0
            for batch_idx,(data,target) in enumerate(pbar):
                data,target = data.to(DEVICE),target.to(DEVICE)

                optimizer.zero_grad()

                y_pred = self.model(data)

                loss = F.nll_loss(y_pred,target)
                
                loss.backward()

                optimizer.step()

                pred = y_pred.argmax(dim=1,keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )
            
            logging.info("Exited the train method of Model Trainer class")
        except Exception as e:
            raise XRayExceptions(e, sys)
    
    def test(self)->None:
        try:
            """
            Description: To test the model.
            Input: model,DEVICE,test_loader
            Output: average loss and accuracy
            """
            logging.info("Entered the test method of ModelTrainer class")

            self.model.eval()

            test_loss: float = 0.0
            correct: int = 0

            with torch.no_grad():
                for (data,target) in self.data_transformation_artifact.transformed_test_object:
                    data,target = data.to(DEVICE),target.to(DEVICE)

                    output = self.model(data)

                    test_loss += F.nll_loss(output,target,reduction="sum").item()

                    pred = output.argmax(dim=1,keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(self.data_transformation_artifact.transformed_test_object.dataset)  

                print(
                    "Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(self.data_transformation_artifact.transformed_test_object.dataset),
                        100.0 * correct / len(self.data_transformation_artifact.transformed_test_object.dataset),
                    )
                )
                logging.info(
                    "Test set: Average loss:{:.4f}, Accuracy: {}/{} {{:.2f}%}".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0 * correct / len(self.data_transformation_artifact.transformed_test_object.dataset),
                    )
                )
                logging.info("Exited the test Method of ModelTrainer class")
        except Exception as e:
            XRayExceptions(e,sys)
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Entered the initiate_model_trainer of ModelTrainer class")

            model:Module = self.model.to(self.model_trainer_config.device)

            optimizer: Optimizer = torch.optim.SGD(
                model.parameters(),
                **self.model_trainer_config.optimizer_params
            )
            scheduler:_LRScheduler = StepLR(
                optimizer=optimizer,
                **self.model_trainer_config.scheduler_params
            )

            for epochs in range(1,self.model_trainer_config.epochs+1):
                print("Epoch :",epochs)

                self.train(optimizer=optimizer)

                optimizer.step()
                scheduler.step()

                self.test()

            os.makedirs(self.model_trainer_config.artifact_dir,exist_ok=True)

            torch.save(model,self.model_trainer_config.trained_model_path)

            train_transform_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=model,
                custom_objects={
                    self.model_trainer_config.train_transforms_key: train_transform_obj
                },
            )

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of ModelTrainer class"
            )
            return model_trainer_artifact;
        except Exception as e:
            XRayExceptions(e,sys)