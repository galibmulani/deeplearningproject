
from Xray.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from Xray.entity.config_entity import ModelEvaluationConfig
from Xray.logger import logging
from Xray.exceptions import XRayExceptions
import sys
from typing import Tuple
from torch import nn
from torch.nn import Module
from torch.optim import SGD,Optimizer
from torch.utils.data import DataLoader
from Xray.ml.model.arch import Net
import torch
from torch.nn import CrossEntropyLoss
from Xray.entity.artifact_entity import ModelEvaluationArtifact
class ModelEvaluation:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_evaluation_config: ModelEvaluationConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
    
    def configuration(self)->Tuple[DataLoader,Module,float,Optimizer]:
        logging.info("Enter the configuration method of ModelEvaluation class")
        try:
            # load test data
            test_loader: DataLoader=(
                self.data_transformation_artifact.transformed_test_object
            )
            # model calling
            model: Module = Net()
            # insert data to model
            model:Module = torch.load(self.model_trainer_artifact.trained_model_path)
            # pass over device
            model.to(self.model_evaluation_config.device)

            # cost function 
            cost: Module  = CrossEntropyLoss()

            #model eval
            model.eval()

            logging.info("Exited the model_evaluation method of ModelEvaluation class")

            return test_loader,model,cost;
        except Exception as e:
            raise XRayExceptions(e,sys)
    def test_net(self) -> float:
        logging.info("Entered the test_net method of Model Evaluation class")
        try:
            test_dataloader,net,cost = self.configuration()
            
            # model inferenece
            with torch.no_grad():
                holder = []

                for _,data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.device)
                    labels = data[1].to(self.model_evaluation_config.device)

                    output = net(images)

                    loss = cost(output,labels)

                    predictions = torch.argmax(output,1)

                    for i in zip(images,labels,predictions):
                        h = list(i)

                        holder.append(h)

                    logging.info(
                        f"Actual_Labels : {labels}  Predictions : {predictions} labels:{loss.item():.4f}")

                    self.model_evaluation_config.test_loss += loss.item()

                    self.model_evaluation_config.test_accuracy +=(
                        (predictions==labels).sum().item()
                    )

                    self.model_evaluation_config.total_batch +=1
                    self.model_evaluation_config.total += labels.size(0)

                    logging.info(
                        f"""Model --> Loss: {self.model_evaluation_config.test_loss/ self.model_evaluation_config.total_batch}
                        Accuracy : {(self.model_evaluation_config.test_accuracy/self.model_evaluation_config.total)*100}%"""
                    )        
                accuracy = (
                    self.model_evaluation_config.test_accuracy
                    / self.model_evaluation_config.total
                )*100

                logging.info("Exited the test_net method of ModelEvaluation Class")

                return accuracy;
        except Exception as e:
            raise XRayExceptions(e,sys)
        
    
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            accuracy = self.test_net()

            model_evaluation_artifact : ModelEvaluationArtifact = (
                ModelEvaluationArtifact(accuracy)
            )

            logging.info("Exited the initiate_model_evaluation method of ModelEvaluation Class")

            return model_evaluation_artifact;
        except Exception as e:
            raise XRayExceptions(e,sys)