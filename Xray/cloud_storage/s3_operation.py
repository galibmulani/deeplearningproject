import os
import sys
from Xray.exceptions import XRayExceptions

class S3Operation:
    def sync_folder_from_s3(self,folder:str,bucket_name:str,folder_bucket_name:str)->None:
        try:
            # aws s3 sync <source> <destination>
            command: str = (
                f'aws s3 sync s3://{bucket_name}/{folder_bucket_name}/ {folder}'
            )
            os.system(command)
        except Exception as e:
            raise XRayExceptions(e, sys)
    
    def sync_folder_to_s3(self,folder:str,bucket_name:str,folder_bucket_name:str)->None:
        try:
            # aws s3 sync <source> <destination>
            command: str= (f'aws s3 sync {folder} s3://{bucket_name}/{folder_bucket_name}/')
            os.system(command)
        except Exception as e:
            raise XRayExceptions(e, sys)