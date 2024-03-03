from light_lm_eval import HarnessModelWrapper
from light_lm_eval.config import EvaluatorConfig
from light_lm_eval.completion import HarnessEvaluatorEngine

harness = HarnessModelWrapper(model_name_or_path="gpt2", model_backend="huggingface")


config = EvaluatorConfig(limit=1)

engine = HarnessEvaluatorEngine(
    model=harness.llm, tasks=["mmlu_flan_n_shot_generative_humanities"], config=config
)

# # Todo: Make get_task_dict a private method
print("=> Task Dict starting ...")
task_dict = engine.get_task_dict()

print("=> Engine starting ...")
results_from_llm = engine.get_result_from_llm(task_dict=task_dict)
