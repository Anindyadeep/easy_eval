import os
import sys
import logging
from typing import List, Optional, Union
from lm_eval import utils
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import eval_logger
from lm_eval.evaluator_utils import run_task_tests


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

        self.task_manager = TaskManager(verbosity=verbosity, include_path=None)

        if tasks is None:
            self.eval_logger.error("Need to specify task to evaluate.")
            sys.exit()

        self.tasks = tasks if isinstance(tasks, list) else [tasks]

    @property
    def list_tasks(self):
        # Todo: Need to give a json where it does an grouping
        self.eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(self.task_manager.all_tasks))
        )
        sys.exit()

    def load(self):
        task_names = self.task_manager.match_tasks(self.tasks)
        for task in [task for task in self.tasks if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in self.tasks if task not in task_names and "*" not in task
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

        return task_names

    def upload_tasks(self, experiment_name: str, provider: str):
        """Upload a task to providers like huggingface / wandb artifcats"""
        raise NotImplementedError()

    def import_task_from_yaml(self, yaml_file):
        """Create or import a new task from a yaml file"""
        raise NotImplementedError

    def import_task_from_huggingface(self, dataset_repo_id: str):
        """Make a huggingface dataset into a task"""
        raise NotImplementedError

    def validate_and_get_task_dict(self) -> dict:
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


if __name__ == "__main__":
    task = HarnessTaskWrapper(tasks=["mmlu"])
    task.list_tasks
