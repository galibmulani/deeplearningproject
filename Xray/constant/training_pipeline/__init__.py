from datetime import datetime
from typing import List

import torch

TIMESTAMP: str = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

#data ingestion constants
ARTIFACTS_DIR: str = "artifacts"
BUCKET_NAME: str = "xraylungimgs2577"
S3_DATA_FOLDER: str = "data"

#data transformation constants
CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

BRIGHTNESS: float = 0.10
CONTRAST: float = 0.10
SATURATION: float = 0.10
HUE: float = 0.10
RESIZE: int = 224
CENTERCROP: int = 224
RANDOMROTATION: int = 10
NORMALIZE_LIST_1: List[int] = [0.485, 0.456, 0.406] # mean
NORMALIZE_LIST_2: List[int] = [0.229, 0.224, 0.225] # std
TRAIN_TRANSFORMS_key: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"
BATCH_SIZE: int = 2
SHUFFLE: bool = False
PIN_MEMORY: bool = True
