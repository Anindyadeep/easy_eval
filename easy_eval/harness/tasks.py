import os
import sys
import logging
from typing import List, Optional, Union
from lm_eval import utils
from lm_eval.utils import (
    eval_logger,
    run_task_tests,
)
import lm_eval.tasks as task_manager 

# TODO: Tasks should not be there inside the constructor 

class HarnessTasks:
    def __init__(self, tasks: Union[str, List[str]], verbosity: Optional[str] = "INFO"):
        """Initiates the task process by loading all the tasks and the necessary utilities.

        Args:
            tasks (Union[str, List[str]]): A List of LM Eval Harness Compatible Task Names
            verbosity (Optional[str], optional): The verbosity level to set. Defaults to "INFO". Set to "DEBUG" to enter debugging mode.
        """
        # Todo:
        # - Set include_path in the main function argument to support additional tasks running from lm_eval_harness backend
        # - task can also be a dir, and we might require to support that too.

        self.eval_logger = utils.eval_logger
        self.eval_logger.setLevel(getattr(logging, f"{verbosity}"))
        self.eval_logger.info(f"Verbosity set to {verbosity}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if tasks is None:
            self.eval_logger.error("Need to specify task to evaluate.")
            sys.exit()

        self.tasks = tasks if isinstance(tasks, list) else [tasks]
        task_manager.initialize_tasks() 

    @property
    def list_tasks(self):
        # Todo: Need to give a json where it does an grouping
        self.eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(task_manager.ALL_TASKS))
        )
        sys.exit()
    
    @property
    def group_tasks(self):
        grouped_task = {}
        for task in self.tasks:
            try:
                grouped_task[task] = self.get_sub_task(main_task=task)
            except Exception as error:
                self.eval_logger.warning(
                    f"Task: {task} does not exist"
                    f"Full error: {error}"
                )
                continue
        return grouped_task
    
    def get_sub_task(self, main_task: str):
        assert main_task in task_manager.ALL_TASKS, ValueError(f"Task {main_task} not found")
        return sorted([task for task in task_manager.ALL_TASKS if task.startswith(main_task)])
    
    def get_dataset_from_task(self, task_limit: int, rank: Optional[int] = 0, world_size: Optional[int] = 1):
        """Fetches the underlined dataset from a given task.
        
        NOTE: This function is bit unstable to use and might give error for some task names. 
        """
        task_dict = task_manager.get_task_dict(self.tasks)
        all_task_data = {}
        
        for task_name, task in task_dict.items():
            if type(task) == tuple:
                _, task = task

            if task is None:
                continue

            if task_limit is not None:
                if task.has_test_docs():
                    task_docs = task.test_docs()
                elif task.has_validation_docs():
                    task_docs = task.validation_docs()
                else:
                    print("Task has neither test_docs nor validation docs")
                    continue
                limit = (
                    int(len(task_docs) * task_limit)
                    if task_limit < 1.0
                    else int(task_limit)
                )
            task.build_all_requests(
                limit=limit, rank=rank, world_size=world_size
            )
            task_wise_data = {"doc_id": [], "prompt": [], "target": []}

            for instance in task.instances:
                task_wise_data["doc_id"].append(instance.doc_id)
                # FIXME: instance.args[0] is a bit explicit and prompt does not makes sense for tasks like hellaswag.
                task_wise_data["prompt"].append(instance.args[0])
                task_wise_data["target"].append(
                    task.doc_to_target(instance.doc)
                )
            all_task_data[task_name] = task_wise_data
        return all_task_data
        
    
    def filter_tasks(self, filter_by: str):
        raise NotImplementedError

    def load(self):
        for task in [task for task in self.tasks if task not in self.tasks]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                self.tasks.append(config)
        task_missing = [
            task for task in self.tasks if task not in self.tasks and "*" not in task
        ]

        if task_missing:
            missing = ", ".join(task_missing)
            self.eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
            )

        return self.tasks 

    def upload_tasks(self, experiment_name: str, provider: str):
        """Upload a task to providers like huggingface / wandb artifcats"""
        raise NotImplementedError()

    def import_task_from_yaml(self, yaml_file):
        """Create or import a new task from a yaml file"""
        raise NotImplementedError

    def import_task_from_huggingface(self, dataset_repo_id: str):
        """Make a huggingface dataset into a task"""
        raise NotImplementedError
