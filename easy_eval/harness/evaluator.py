import torch
import random
import numpy as np
from typing import Optional, List, Union, Tuple 

import lm_eval.api
from lm_eval.utils import simple_parse_args_string, eval_logger, get_git_commit_hash

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
        """Evaluator engine compatible with lm-eval-harness

        Args:
            model (Union[HarnessModels, lm_eval.api.model.LM, lm_eval.api.model.CachingLM]): Harness compatible model. 
            config (Optional[EvaluatorConfig], optional): A EvaluatorConfig which helps to set different generation configurations. Defaults to EvaluatorConfig().

        Raises:
            TypeError: If model is not based on the given types. Since it is not compatible with the 
            evaluator engine. 
        """
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
    
    def evaluate_from_model_response(
        self, 
        tasks: Union[List[str], List[HarnessTask]],
        metadata: dict, 
        gen_kwargs: Optional[Union[str, dict]] = None,
        bootstrap_iters: Optional[int] = 100000
    ) -> dict:
        """Evaluate results from model responses 

        Args:
            tasks (Union[List[str], List[HarnessTask]]): A List of tasks that needs to evaluated 
            metadata: (dict): Metadata consists for the task information 
            gen_kwargs (Optional[Union[str, dict]], optional): generation kwargs for the model. Defaults to None.
            bootstrap_iters (Optional[int], optional): Defaults to 100000.

        Returns:
            dict: Results for each of the tasks 
        """
        assert tasks is None, ValueError("tasks can not be None.")
        
        if gen_kwargs is not None:
            gen_kwargs = simple_parse_args_string(gen_kwargs)
            eval_logger.warning(
                "generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks."
            )
            if gen_kwargs == "":
                gen_kwargs = None
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.numpy_random_seed)
        torch.manual_seed(self.config.torch_random_seed)
        
        task_dict = HarnessTaskManager.get_task_dict(tasks=tasks, num_fewshot=self.config.num_fewshot, gen_kwargs=gen_kwargs)
        
        results = harness_postprocessor(
            lm=self.lm, task_dict=task_dict, metadata=metadata, config=self.config, bootstrap_iters=bootstrap_iters
        )
        
        if self.lm.rank == 0:
            model_name = self.lm.config._name_or_path

            # add info about the model and few shot config
            results["config"] = {
                "model": model_name,
                "batch_size": self.config.batch_size,
                "batch_sizes": list(self.lm.batch_sizes.values())
                if hasattr(self.lm, "batch_sizes")
                else [],
                "device": self.lm.device,
                "use_cache": self.config.use_cache,
                "limit": self.config.limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
            }
            results["git_hash"] = get_git_commit_hash()
            return results
        
    def evaluate(
        self, 
        tasks: Union[List[str], List[HarnessTask]], 
        gen_kwargs: Optional[Union[str, dict]] = None,
        bootstrap_iters: Optional[int] = 100000
    ) -> Tuple[dict, list]:
        """Function to evaluate model on the specified tasks using lm-eval-harness backend

        Args:
            tasks (Union[List[str], List[HarnessTask]]): A List of tasks that needs to evaluated 
            gen_kwargs (Optional[Union[str, dict]], optional): generation kwargs for the model. Defaults to None.
            bootstrap_iters (Optional[int], optional): Defaults to 100000.


        Returns:
            Tuple[dict, list]: Returns the result and the raw responses from the model
        """
        
        if isinstance(self.lm, HarnessModels):
            raw_responses, _, metadata = self.lm.generate(
                tasks=tasks, return_task_metadata=True
            )
        else:
            raw_responses, _, metadata = _build_tasks_and_generate(
                tasks=tasks, lm=self.lm, limit=self.config.limit, write_out=self.config.write_out
            ) 
        
        results = self.evaluate_from_model_response(
            tasks=tasks,
            metadata=metadata, gen_kwargs=gen_kwargs, bootstrap_iters=bootstrap_iters
        )
        return results, raw_responses