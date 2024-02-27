import collections
import itertools
import logging
import random
from typing import Optional, Union, List

import numpy as np
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.logging_utils import add_env_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    eval_logger,
    run_task_tests,
    simple_parse_args_string,
)


from light_lm_eval.config import EvaluatorConfig


class HarnessEvaluatorEngine:
    def __init__(
        self,
        model,
        tasks: List[str],
        task_manager=None,
        config: Optional[EvaluatorConfig] = EvaluatorConfig,
    ) -> None:

        eval_logger.setLevel(getattr(logging, f"{config.verbosity}"))
        if config.seed is not None:
            eval_logger.info(f"Setting random seed to {config.seed}")
            random.seed(config.seed)

        if config.numpy_seed is not None:
            eval_logger.info(f"Setting numpy seed to {config.numpy_seed}")
            np.random.seed(config.numpy_seed)

        if config.torch_seed is not None:
            eval_logger.info(f"Setting torch manual seed to {config.torch_seed}")
            torch.manual_seed(config.torch_seed)

        if tasks is None:
            tasks = []
        assert (
            tasks != []
        ), "No tasks specified, or no tasks found. Please verify the task names."

        if config.gen_kwargs is not None:
            self.gen_kwargs = simple_parse_args_string(config.gen_kwargs)
            eval_logger.warning(
                "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
            )
            if config.gen_kwargs == "":
                self.gen_kwargs = None

        if config.use_cache is not None:
            print(f"Using cache at {config.use_cache + '_rank' + str(lm.rank) + '.db'}")
            lm = lm_eval.api.model.CachingLM(
                lm,
                config.use_cache
                # each rank receives a different cache db.
                # necessary to avoid multiple writes to cache at once
                + "_rank" + str(lm.rank) + ".db",
            )

        self.task_manager = (
            task_manager if task_manager is not None else TaskManager(config.verbosity)
        )

        self.lm = model
        self.config = config
        self.tasks = tasks

        self.versions = collections.defaultdict(dict)
        self.task_hierarchy = collections.defaultdict(list)
        self.configs = collections.defaultdict(dict)
        self.num_fewshot = collections.defaultdict(int)
        self.results = collections.defaultdict(dict)
        self.requests = collections.defaultdict(list)
        self.vals = collections.defaultdict(list)

        if self.config.predict_only:
            self.log_samples = True

    def get_task_dict(self) -> dict:
        task_dict = get_task_dict(self.tasks, self.task_manager)

        for task_name in task_dict.keys():
            task_obj = task_dict[task_name]
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
                if task_obj is None:
                    continue

            if task_obj.get_config("output_type") == "generate_until":
                if self.config.gen_kwargs is not None:
                    task_obj.set_config(
                        key="generation_kwargs",
                        value=self.config.gen_kwargs,
                        update=True,
                    )

                if self.config.predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

            if self.config.num_fewshot is not None:
                if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                    eval_logger.info(
                        f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                    )
                else:
                    eval_logger.warning(
                        f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {self.config.num_fewshot}"
                    )
                    task_obj.set_config(
                        key="num_fewshot", value=self.config.num_fewshot
                    )
        if self.config.check_integrity:
            run_task_tests(self.tasks)

        return task_dict

    def _get_each_request_list(self, task_dict: dict):
        for task_name, task in task_dict.items():
            if isinstance(task, tuple):
                group_name, task = task
                self.task_hierarchy[group_name].append(task_name)
                self.versions[group_name] = "N/A"

            else:
                group_name = None
                self.task_hierarchy[task_name] = []

            if task is None:
                continue

            self.versions[task_name] = task.VERSION
            self.configs[task_name] = dict(task.dump_config())

            # Number of few-shots for printing.
            if (n_shot := self.configs[task_name].get("num_fewshot")) == 0:
                n_shot = (
                    self.configs[task_name].get("metadata", {}).get("num_fewshot", 0)
                )
            self.num_fewshot[task_name] = n_shot

            if "task_alias" in self.configs[task_name]:
                self.results[task_name]["alias"] = self.configs[task_name]["task_alias"]

            if (
                ("group_alias" in self.configs[task_name])
                and (group_name not in self.results)
                and (group_name is not None)
            ):
                self.results[group_name]["alias"] = self.configs[task_name][
                    "group_alias"
                ]

            if self.config.limit is not None:
                if task.has_test_docs():
                    task_docs = task.test_docs()
                elif task.has_validation_docs():
                    task_docs = task.validation_docs()
                else:
                    raise RuntimeError("Task has neither test_docs nor validation_docs")
                self.limit = (
                    int(len(task_docs) * self.config.limit)
                    if self.config.limit < 1.0
                    else int(self.config.limit)
                )

            task.build_all_requests(
                limit=self.limit, rank=self.lm.rank, world_size=self.lm.world_size
            )

            eval_logger.debug(
                f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
            )

            # Not using writeout here, skipping the prompt printing part

            for instance in task.instances:
                reqtype = instance.request_type
                self.requests[reqtype].append(instance)

            if self.lm.world_size > 1:
                instances_rnk = torch.tensor(
                    len(task._instances), device=self.lm.device
                )
                gathered_item = (
                    self.lm.accelerator.gather(instances_rnk)
                    .cpu()
                    .detach()
                    .numpy()
                    .tolist()
                )

                # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
                numpad = max(gathered_item) - gathered_item[self.lm.rank]
                self.padding_requests[task.OUTPUT_TYPE] += numpad
                self.config.limit = self.limit

    def get_result_from_llm(
        self,
        task_dict,
        verbosity: str = "INFO",
    ):
        # we are returning the config along with requests, since some config like limit is overwritten
        # similarly more config might be overwritten and we might need to support that too

        eval_logger.setLevel(getattr(logging, f"{verbosity}"))
        self._get_each_request_list(task_dict=task_dict)

        for reqtype, reqs in self.requests.items():
            eval_logger.info(f"Running {reqtype} requests")
            # create `K` copies of each request `req` based off `K = req.repeats`
            cloned_reqs = []
            for req in reqs:
                cloned_reqs.extend([req] * req.repeats)

            if (self.lm.world_size > 1) and (self.padding_requests[reqtype] > 0):
                for _ in range(self.padding_requests[reqtype]):
                    cloned_reqs.extend([req] * req.repeats)

            # run requests through model
            resps = getattr(self.lm, reqtype)(cloned_reqs)

            # put responses from model into a list of length K for each request.
            for x, req in zip(resps, cloned_reqs):
                req.resps.append(x)

            if self.lm.world_size > 1:
                self.lm.accelerator.wait_for_everyone()
        return self.requests, self.config
