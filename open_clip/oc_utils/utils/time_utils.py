import time
from datetime import datetime

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = datetime.now()
        result = original_fn(*args, **kwargs)
        end_time = datetime.now()
        print("WorkingTime[{}]: {} ".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn