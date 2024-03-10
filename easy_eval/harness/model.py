import os
import logging
import importlib
from typing import Optional, List, Union

import lm_eval
from lm_eval import utils
from lm_eval.utils import eval_logger

from easy_eval.config import EvaluatorConfig
from easy_eval.harness.tasks import HarnessTask
from easy_eval.harness.utils import _handle_non_serializable, _build_tasks_and_generate


class HarnessModels:
    def __init__(
        self,
        model_name_or_path: str,
        model_backend: str,
        verbosity: Optional[str] = "INFO",
        config: Optional[EvaluatorConfig] = EvaluatorConfig(), 
        **kwargs,
    ) -> None:
        """Harness Models are the LLMs running on various inference engines but strictly compatible with lm-eval-harness's ecosystem"""
        _model_classfile_mapper = {
            "huggingface": ("huggingface", "HFLM"),
            "vllm": ("vllm_causallms", "VLLM"),
            "anthropic": ("anthropic_llms", "AnthropicLM"),
            "gguf": ("gguf", "GGUFLM"),
            "ggml": ("gguf", "GGUFLM"),
            "mamba": ("mamba_lm", "MambaLMWrapper"),
            "openvino": ("optimum_lm", "OptimumLM"),
            "openai": ("openai_completions", "OpenaiCompletionsLM"),
        }

        _available_backends = list(_model_classfile_mapper.keys())

        assert model_backend in _available_backends, ValueError(
            f"model_backend: {model_backend} does not exist. Available backends: {_available_backends}"
        )
        _llm_module = getattr(
            importlib.import_module(
                f"lm_eval.models.{_model_classfile_mapper[model_backend][0]}"
            ),
            _model_classfile_mapper[model_backend][1],
        )
        
        self.config = config
        model = _llm_module(model_name_or_path, **kwargs)
        
        if self.config.use_cache is not None:
            self.lm = lm_eval.api.model.CachingLM(
                model, self.config.use_cache + "_rank" + str(model.rank) + ".db"
            )    
            eval_logger.info(
                f"Using cache at {self.use_cache + '_rank' + str(model.rank) + '.db'}"
            )
        else:
            self.lm = model 
        

        # other stuffs
        self.verbosity = verbosity
        self.eval_logger = utils.eval_logger
        self.eval_logger.setLevel(getattr(logging, f"{verbosity}"))
        self.eval_logger.info(f"Verbosity set to {verbosity}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def generate(
        self, 
        tasks: Union[List[str], HarnessTask], 
        return_task_metadata: bool = False
    ) -> List[str]:
        
        responses, task_dict, task_metadata = _build_tasks_and_generate(
            tasks=tasks, lm=self.lm, limit=self.config.limit, write_out=self.config.write_out
        )
        return (responses, task_dict, task_metadata) if return_task_metadata else responses

