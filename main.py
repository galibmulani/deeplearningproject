from Xray.pipeline.training_pipeline import TrainingPipeline
from Xray.exceptions import XRayExceptions
import sys

def start_training_pipeline():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        raise XRayExceptions(e, sys) 
    
if __name__ == "__main__":
    start_training_pipeline()