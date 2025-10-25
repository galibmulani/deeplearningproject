import os
import sys
from Xray.exceptions import XRayExceptions

class S3Operation:
    def sync_folder_from_s3()->None:
        try:
            pass
        except Exception as e:
            raise XRayExceptions(e, sys)