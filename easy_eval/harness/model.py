import re
import os
import json
import logging
import numpy as np
import importlib
from pathlib import Path
from typing import Optional, List

from lm_eval import evaluator, utils
from lm_eval.utils import make_table
from easy_eval.harness.tasks import HarnessTasks
from easy_eval.config import EvaluatorConfig


# Todo: Make more indepedent component which can evaluate with LLM's output only without requiring LLMs.
# Todo: Find and make as many detachable components as possible


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


class HarnessEvaluator:
    def __init__(
        self,
        model_name_or_path: str,
        model_backend: str,
        verbosity: Optional[str] = "INFO",
        **kwargs,
    ) -> None:
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
        self.llm = _llm_module(model_name_or_path, **kwargs)

        # other stuffs
        self.verbosity = verbosity
        self.eval_logger = utils.eval_logger
        self.eval_logger.setLevel(getattr(logging, f"{verbosity}"))
        self.eval_logger.info(f"Verbosity set to {verbosity}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _initialise_evaluation(self, tasks: List[str]) -> None:
        self.task_wrapper = HarnessTasks(tasks=tasks, verbosity=self.verbosity)
        self.tasks = self.task_wrapper.load()
        self.eval_logger.info(f"Evaluating for tasks: {self.tasks}")
        print("=> Done")

    def evaluate(
        self,
        tasks: List[str],
        config: Optional[EvaluatorConfig] = EvaluatorConfig(),
        show_results_terminal: Optional[bool] = False,
    ):
        self._initialise_evaluation(tasks=tasks)

        if config.output_path:
            path = Path(config.output_path)
            if (
                path.is_file()
                or Path(config.output_path).joinpath("results.json").is_file()
            ):
                self.eval_logger.warning(
                    f"File already exists at {path}. Results will be overwritten."
                )
                output_path_file = path.joinpath("results.json")
            elif path.suffix in (".json", ".jsonl"):
                output_path_file = path
                path.parent.mkdir(parents=True, exist_ok=True)
                path = path.parent
            else:
                path.mkdir(parents=True, exist_ok=True)
                output_path_file = path.joinpath("results.json")

        print(f"=> Starting to evaluate on tasks: {self.tasks}")
        evaluation_results = evaluator.simple_evaluate(
            model=self.llm,
            tasks=self.tasks,
            num_fewshot=config.num_fewshot,
            batch_size=config.batch_size,
            max_batch_size=config.max_batch_size,
            device=config.device,
            use_cache=config.use_cache,
            limit=config.limit,
            decontamination_ngrams_path=config.decontamination_ngrams_path,
            check_integrity=config.check_integrity,
            write_out=config.write_out,
            log_samples=config.log_samples,
            gen_kwargs=config.gen_kwargs,
            task_manager=self.task_wrapper.task_manager,
            predict_only=config.predict_only,
            random_seed=config.seed,
            numpy_random_seed=config.numpy_seed,
            torch_random_seed=config.torch_seed,
        )

        # Todo: Need wandb logging

        if evaluation_results is not None:
            if config.log_samples:
                samples = evaluation_results.pop("samples")

            dumped = json.dumps(
                evaluation_results,
                indent=2,
                default=_handle_non_serializable,
                ensure_ascii=False,
            )

        if config.output_path:
            output_path_file.open("w", encoding="utf-8").write(dumped)

            if config.log_samples:
                for task_name, config in evaluation_results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", evaluation_results.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        if show_results_terminal:
            print(make_table(evaluation_results))
        if "groups" in evaluation_results:
            print(make_table(evaluation_results, "groups"))
        return evaluation_results
