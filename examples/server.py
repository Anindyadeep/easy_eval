# Todo: Make a evaluation server
# Where while setting up the server the user can
# - set their model of choice
# - set their custom tasks of choice 
# In the client side of things user will have the following endpoints
# - /GET:  metadata (contain the server config like model name etc etc)
# - /GET:  available_tasks (return all the available tasks)
# - /POST: evaluate model

from fastapi import FastAPI
from easy_eval.model import HarnessEvaluator
from easy_eval.tasks import HarnessTasks

class EvaluationServer:
    raise NotImplementedError