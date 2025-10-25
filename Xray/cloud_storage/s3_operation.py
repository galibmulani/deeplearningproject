import os
import sys
from Xray.exceptions import XRayExceptions

def function_that_raises():
    try:
        1/0
    except Exception as e:
        raise XRayExceptions(e,sys)
    
def function_that_raises_convert_error():
    try:
        int("abc")
    except Exception as e:
        raise XRayExceptions(e,sys)
    
#function_that_raises()
print()
#function_that_raises_convert_error()
for fun in [function_that_raises, function_that_raises_convert_error]:
    try:
        fun()
    except XRayExceptions as xe:
        print(xe)