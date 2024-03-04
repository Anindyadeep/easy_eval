import os
import logging
from typing import List, Optional, Union
from lm_eval import utils
import lm_eval.tasks as task_manager
from dataclasses import dataclass

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
    task = None
    loaded_yaml_file: Optional[str]
    csv_file_name: Optional[str]
    huggingface_dataset_name: Optional[str]

    _is_loaded_from_file_or_repo: Optional[bool]
    _is_loaded_from_yaml: Optional[bool]
    _is_loaded_from_huggingface_datasets: Optional[bool]
    _is_loaded_from_csv_file: Optional[bool]

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
        if not self.is_loaded_from_yaml:
            assert self.task in task_manager.ALL_TASKS, ValueError(
                f"Task {self.task} not found"
            )

        sub_tasks = sorted(
            [task for task in task_manager.ALL_TASKS if task.startswith(self.task)]
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
                limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)
            task.build_all_requests(limit=limit, rank=rank, world_size=world_size)
            task_wise_data = {"doc_id": [], "prompt": [], "target": []}

            for instance in task.instances:
                task_wise_data["doc_id"].append(instance.doc_id)
                # TODO: instance.args[0] is a bit explicit and prompt does not makes sense for tasks like hellaswag.
                task_wise_data["prompt"].append(instance.args[0])
                task_wise_data["target"].append(task.doc_to_target(instance.doc))
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

        task_missing = [task for task in tasks if task not in tasks and "*" not in task]

        if task_missing:
            missing = ", ".join(task_missing)
            cls.eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
            )

        else:
            for task in tasks:
                if not os.path.isfile(task):
                    loaded_tasks.append(HarnessTask(name=task))
        return loaded_tasks
