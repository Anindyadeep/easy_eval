# EasyEval

EasyEval is a fully open-source evaluation wrapper that aims to streamline the integration, customization, and expansion of robust evaluation engines like [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-eval-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) into existing production-grade or research pipelines effortlessly. It supports over 200 existing datasets and can be easily adapted for custom ones, making it a versatile solution for enhancing evaluation processes.

### But Why?

Evaluation has been open-problem for LLMs. When evaluating LLMs into production, we need to rely on different evaluation techniques. However the problem that we lot of times face is to integrate good evaluation engines into different existing production LLM pipelines. 

So what are the solutions:

1. Either go for an enterprise solution.
2. Or look for Open Source solutions. 

Now there are some handful of open-soure libraries that does evaluation on large scale evaluation datasets. Some of the examples are:

1. [LM Evaluation Harness by Eleuther AI](https://github.com/EleutherAI/lm-evaluation-harness)
2. [BigCode Evaluation Harness by the BigCode Project](https://github.com/bigcode-project/bigcode-evaluation-harness)
3. [Stanford HELM](https://crfm.stanford.edu/helm/lite/latest/)
4. [OpenCompass](https://opencompass.org.cn/home)

Other than that we have tons and tons of evaluation libraries where a huge percentage is an extension of the above engines. The way this engine works they define some taxonomy of how they evaluate. 

For example: LM Evaluation Harness by Eleuther AI defines different tasks and under each task we have different datasets. We use the "test/evaluation" split of the datasets to evaluate the LLM of choice. 

### The problem

The problem with these evaluators is, most of them are CLI first. They expose very little documentation on their actual API interfaces. These libraries becomes super useful if they can be easily integrated or extended or customized with newer tasks in existing production pipelines. Production pipelines like:

1. Making evaluation REST-API servers
2. CI/CD pipelines for evaluation for LLM fine-tuning
3. Leaderboard generations to compare across checkpoints or different LLMs.
4. Supporting any custom model or engine. Example TensorRT or any API endpoint.
5. GPT as evalutor etc. 

And like this many more. 

## The objective of the Library

This library acts as a wrapper to combine both the engines lm-eval-harness (mostly consist of evaluation dataset across different general tasks) and bigcode-eval-harness (evaluation dataset exclusivelty for code-generation tasks) with common interfaces. The features of the library include:

1. Adding a common interface between the two libraries for handling evaluation workloads. 
2. Providing interfaces to solve the above problems. 
3. Cutomization of models / addition of new benchmark datasets. 

## Getting Started and Usage:

Let's get started to install the library first. To do that open the terminal and make new virtual environment, and intall easyeval. 

```
pip install easyeval
```

### Usage

The very first version include a simple interface to interact with lm-eval-harness engine. Here is how you can do that. 

```python
from easy_eval import HarnessEvaluator
from easy_eval.config import EvaluatorConfig
```

Evaluation Config is where you provide your model's generation configuration. You can checkout all the configs [here](/easy_eval/config.py). After this, we instantiate our evaluator. 

```python
harness = HarnessEvaluator(model_name_or_path="gpt2", model_backend="huggingface", device="cpu")

# For device you can set cpu or cuda, the standard way of setting up devices. 
```

`HarnessEvaluator` expects you to provide the `model_backend`. Here are some supported backends:

1. [HuggingFace](https://huggingface.co/)
2. [vLLM](https://github.com/vllm-project/vllm)
3. [Anthropic](https://www.anthropic.com/)
4. [OpenAI](https://platform.openai.com/docs/introduction)
5. [OpenVino](https://github.com/openvinotoolkit/openvino)
6. [GGML/GGUF](https://github.com/ggerganov/ggml)
7. [Mamba](https://github.com/mamba-org/mamba)

And also `model_name_or_path` which is the name the model (if huggingface repo) or the model path of the corresponding `model_backend`

Once we instantiated our evaluator, we are going to define our config. Defining config is fully optional. If we not pass config, the default values in config will be choosen. 

```python
config = EvaluatorConfig(
    limit=10 # the number of datapoints to take for evaluation
)
```

And now we get our evaluation result by passing the config and list of evaluation tasks, we want our model to evaluate on. 

```python
results = harness.evaluate(
    tasks=["babi"],
    config=config, show_results_terminal=True
)

print(results)
```

This will return a `result` in a json format.

## Contributing

`easyeval` is at super early stage right now. You can check out the [roadmap](https://github.com/Anindyadeep/easy_eval/issues/2) to see what are the expected features to come in future. 

This is a fully open-sourced project. So contributions are highly appreciated. Here is how you can contribute:

1. Open issues to suggest improvement or features.
2. You can contribute to existing issues or do bug fixing by adding a pull request.


## Reference and Citations 

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```

```
@misc{bigcode-evaluation-harness,
  author       = {Ben Allal, Loubna and
                  Muennighoff, Niklas and
                  Kumar Umapathi, Logesh and
                  Lipkin, Ben and
                  von Werra, Leandro},
  title = {A framework for the evaluation of code generation models},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bigcode-project/bigcode-evaluation-harness}},
  year = 2022,
}
```