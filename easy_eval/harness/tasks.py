import os
import logging
from typing import List, Optional, Union, Any

import torch
import lm_eval
from lm_eval import utils
import lm_eval.tasks as task_manager
from dataclasses import dataclass
from lm_eval.utils import eval_logger

import collections

task_manager.initialize_tasks()


# TODO: Add documentaion on how to use each and every function
class classproperty:
    def __init__(self, func):
        self.func_get = func

    def __get__(self, instance, owner):
        return self.func_get(owner)


@dataclass
class HarnessTask:
    name: str
    task: Any = None
    loaded_yaml_file: Optional[str] = None
    csv_file_name: Optional[str] = None
    huggingface_dataset_name: Optional[str] = None

    _is_loaded_from_file_or_repo: Optional[bool] = False
    _is_loaded_from_yaml: Optional[bool] = False
    _is_loaded_from_huggingface_datasets: Optional[bool] = False
    _is_loaded_from_csv_file: Optional[bool] = False

    def __init__(self, name: str):
        self.name = name
        if not self._is_loaded_from_file_or_repo:
            self.task = name

    def load_from_csv(self, csv_file: str):
        self._is_loaded_from_file_or_repo = True
        self._is_loaded_from_csv_file = True
        raise NotImplementedError

    def load_from_huggingface(self, dataset_repo: str):
        self._is_loaded_from_file_or_repo = True
        self._is_loaded_from_huggingface_datasets = True
        raise NotImplementedError

    def load_from_yaml(self, yaml_config_file: str):
        # TODO: research more about this since this gives areas to add new files.
        assert os.path.isfile(yaml_config_file), FileNotFoundError(
            f"task: {yaml_config_file} is not a file"
        )
        config = utils.load_yaml_config(yaml_config=yaml_config_file)
        self._is_loaded_from_file_or_repo = True
        self._is_loaded_from_yaml = True
        self.task = config

    @property
    def get_sub_tasks(self):
        if not self._is_loaded_from_file_or_repo:
            assert self.task in task_manager.ALL_TASKS, ValueError(
                f"Task {self.task} not found"
            )

        sub_tasks = sorted(
            [
                task
                for task in task_manager.ALL_TASKS
                if task.startswith(self.task)
            ]
        )
        return {
            "main_task": self.task,
            "num_sub_tasks": len(sub_tasks),
            "sub_tasks": sub_tasks,
        }

    def upload_to_hf_hub(self):
        raise NotImplementedError

    def get_dataset(
        self, limit: int, rank: Optional[int] = 0, world_size: Optional[int] = 1
    ):
        """Fetches the underlined dataset from a given task.

        NOTE: This function is bit unstable to use and might give error for some task names.
        """
        task_dict = task_manager.get_task_dict(self.task)
        all_task_data = {}

        for task_name, task in task_dict.items():
            if type(task) == tuple:
                _, task = task

            if task is None:
                continue

            if limit is not None:
                if task.has_test_docs():
                    task_docs = task.test_docs()
                elif task.has_validation_docs():
                    task_docs = task.validation_docs()
                else:
                    print("Task has neither test_docs nor validation docs")
                    continue
                limit = (
                    int(len(task_docs) * limit) if limit < 1.0 else int(limit)
                )
            task.build_all_requests(
                limit=limit, rank=rank, world_size=world_size
            )
            task_wise_data = {"doc_id": [], "prompt": [], "target": []}

            for instance in task.instances:
                task_wise_data["doc_id"].append(instance.doc_id)
                # TODO: instance.args[0] is a bit explicit and prompt does not makes sense for tasks like hellaswag.
                task_wise_data["prompt"].append(instance.args[0])
                task_wise_data["target"].append(
                    task.doc_to_target(instance.doc)
                )
            all_task_data[task_name] = task_wise_data
        return all_task_data


class HarnessTaskManager:
    verbosity = "INFO"
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    eval_logger.info(f"Verbosity set to {verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    @classproperty
    def list_all_tasks(cls):
        all_tasks = task_manager.ALL_TASKS
        return {"num_tasks": len(all_tasks), "tasks": all_tasks}

    @classmethod
    def load_tasks(cls, tasks: Union[str, List[str]]):
        loaded_tasks = []
        for task in [task for task in tasks if task not in tasks]:
            if os.path.isfile(task):
                loaded_tasks.append(HarnessTask(name=task).load_from_yaml(task))

        task_missing = [
            task for task in tasks if task not in tasks and "*" not in task
        ]

        if task_missing:
            missing = ", ".join(task_missing)
            cls.eval_logger.error(f"Tasks were not found: {missing}\n")
            raise ValueError(f"Tasks not found: {missing}")

        else:
            for task in tasks:
                if not os.path.isfile(task):
                    loaded_tasks.append(HarnessTask(name=task))
        return loaded_tasks

    @classmethod
    def get_task_dict(
        cls,
        tasks: Union[List[str], List[HarnessTask]],
        num_fewshot: Optional[int] = None,
        gen_kwargs: Optional[dict] = None,
    ):
        tasks = [
            task.task if isinstance(task, HarnessTask) else task
            for task in tasks
        ]
        task_dict = lm_eval.tasks.get_task_dict(tasks)

        for task_name in task_dict.keys():
            task_obj = task_dict[task_name]
            if isinstance(task_obj, tuple):
                group, task_obj = task_obj
                if task_obj is None:
                    continue

            config = task_obj._config
            if (
                config["output_type"] == "generate_until"
                and gen_kwargs is not None
            ):
                config["generation_kwargs"].update(gen_kwargs)

            if num_fewshot is not None:
                if config["num_fewshot"] == 0:
                    eval_logger.info(
                        f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                    )
                else:
                    default_num_fewshot = config["num_fewshot"]
                    if default_num_fewshot:
                        # warn a user, if a specific num_fewshot > 0 was specified.
                        # if unspecified in config, no warning message
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )

                    task_obj._config["num_fewshot"] = num_fewshot
        return task_dict

    @classmethod
    def build_task_requests(
        cls,
        lm: Any,
        task_dict: dict,
        limit: int = None,
        write_out: bool = False,
    ):
        """Builds request and metadata from task dict
        Here is what the metadata contains info about:
            results: stores the final result for each task, for each metric/filter pair.
            versions: tracks each task's version.
            configs: tracks the YAML configs of all chosen tasks.
            samples: logs info about each document evaluated.
            requests: tracks all instances/requests a model must generate output on.
            results_agg: aggregated task scores presented with groups.
            groups_agg: aggregated groups scores only.
            padding_requests: stores the amount to pad out requests per request type so that
                            the number of forward passes per distributed rank is equal.
            task_hierarchy: stores the hierarchy to do proper ordering.
            num_fewshot: stores the num-fewshot value per task.
        Returns:
            task_dict, metadata
        """
        metadata = {
            "results": collections.defaultdict(dict),
            "versions": collections.defaultdict(dict),
            "configs": collections.defaultdict(dict),
            "samples": collections.defaultdict(list),
            "requests": collections.defaultdict(list),
            "results_agg": collections.defaultdict(dict),
            "groups_agg": collections.defaultdict(dict),
            "padding_requests": collections.defaultdict(int),
            "task_hierarchy": collections.defaultdict(list),
            "num_fewshot": collections.defaultdict(int),
        }

        for task_name, task in task_dict.items():
            if isinstance(task, tuple):
                group_name, task = task
                metadata["task_hierarchy"][group_name].append(task_name)
                metadata["versions"][group_name] = "N/A"
            else:
                group_name = None
                metadata["task_hierarchy"][task_name] = []

            if task is None:
                continue

            metadata["versions"][task_name] = task.VERSION
            metadata["configs"][task_name] = dict(task.dump_config())

            if "num_fewshot" in metadata["configs"][task_name]:
                n_shot = metadata["configs"][task_name]["num_fewshot"]
            else:
                n_shot = 0
            metadata["num_fewshot"][task_name] = n_shot

            if "task_alias" in metadata["configs"][task_name]:
                metadata["results"][task_name]["alias"] = metadata["configs"][
                    task_name
                ]["task_alias"]

            if (
                ("group_alias" in metadata["configs"][task_name])
                and (group_name not in metadata["results"])
                and (group_name is not None)
            ):
                metadata["results"][group_name]["alias"] = metadata["configs"][
                    task_name
                ]["group_alias"]

            if limit is not None:
                if task.has_test_docs():
                    task_docs = task.test_docs()
                elif task.has_validation_docs():
                    task_docs = task.validation_docs()
                else:
                    raise RuntimeError(
                        "Task has neither test_docs nor validation_docs"
                    )
                limit = (
                    int(len(task_docs) * limit) if limit < 1.0 else int(limit)
                )

            task.build_all_requests(
                limit=limit, rank=lm.rank, world_size=lm.world_size
            )

            eval_logger.debug(
                f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
            )

            if write_out:
                for inst in task.instances:
                    if inst.doc_id < 1:
                        eval_logger.info(
                            f"Task: {task_name}; document {inst.doc_id} context prompt (starting on next line):"
                            f"\n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
                        )
                        eval_logger.info(f"Request: {str(inst)}")

            for instance in task.instances:
                reqtype = instance.request_type
                metadata["requests"][reqtype].append(instance)

            if lm.world_size > 1:
                instances_rnk = torch.tensor(
                    len(task._instances), device=lm.device
                )
                gathered_item = (
                    lm.accelerator.gather(instances_rnk)
                    .cpu()
                    .detach()
                    .numpy()
                    .tolist()
                )

                # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
                numpad = max(gathered_item) - gathered_item[lm.rank]
                metadata["padding_requests"][task.OUTPUT_TYPE] += numpad

        return task_dict, metadata
