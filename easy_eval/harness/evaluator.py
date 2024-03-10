import torch
import random
import numpy as np
from typing import Optional, List, Union 

import lm_eval.api
from lm_eval.utils import simple_parse_args_string, eval_logger

from easy_eval.config import EvaluatorConfig
from easy_eval.harness.tasks import HarnessTask, HarnessTaskManager
from easy_eval.harness.model import HarnessModels
from easy_eval.harness.utils import harness_postprocessor, _build_tasks_and_generate

# TODO: Add good documentation for each of the function     
        

class HarnessEvaluator:
    def __init__(
        self, 
        model: Union[HarnessModels, lm_eval.api.model.LM, lm_eval.api.model.CachingLM], 
        config: Optional[EvaluatorConfig] = EvaluatorConfig()
    ):
        self.config = config
        if isinstance(model, HarnessModels):
            self.lm = model.lm
        elif isinstance(model, lm_eval.api.model.LM) or isinstance(lm_eval.api.model.CachingLM):
            self.lm = model
        else:
            raise TypeError(
                "model should be strictly either type HarnessModels or lm_eval.api.model.LM",
                f"but found {type(model)}"
            )
    
    def simple_evaluate(
        self, 
        tasks: Union[List[str], List[HarnessTask]], 
        gen_kwargs: Optional[Union[str, dict]] = None
    ):
        random.seed(self.config.random_seed)
        np.random.seed(self.config.numpy_random_seed)
        torch.manual_seed(self.config.torch_random_seed) 
        
        assert tasks is None, ValueError("tasks can not be None.")
        
        if gen_kwargs is not None:
            gen_kwargs = simple_parse_args_string(gen_kwargs)
            eval_logger.warning(
                "generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks."
            )
            if gen_kwargs == "":
                gen_kwargs = None
        
        self.task_dict = HarnessTaskManager.get_task_dict(tasks=tasks, num_fewshot=self.config.num_fewshot, gen_kwargs=gen_kwargs)
        
        results, raw_responses = self.evaluate(task_dict=self.task_dict)
        
        raise NotImplementedError(
            "TODO: Postprocessing of results left"
        )
    
    
    def evaluate(self, tasks: Union[List[str], List[HarnessTask]]):
        if isinstance(self.lm, HarnessModels):
            raw_responses, task_dict, metadata = self.lm.generate(
                tasks=tasks, return_task_metadata=True
            )
        else:
            raw_responses, task_dict, metadata = _build_tasks_and_generate(
                tasks=tasks, lm=self.lm, limit=self.config.limit, write_out=self.config.write_out
            ) 
        
        results = harness_postprocessor(
            lm=self.lm, task_dict=task_dict, metadata=metadata, config=self.config
        )
        return results, raw_responses