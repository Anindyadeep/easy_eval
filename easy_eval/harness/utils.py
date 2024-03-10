# TODO: This needs iteration with latest code 
# Lot of parts of this part of code is solely taken from lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py

import itertools
import collections
from typing import Union, List

import torch 
import lm_eval
import numpy as np 

from easy_eval.config import EvaluatorConfig
from easy_eval.harness.tasks import HarnessTaskManager, HarnessTask
from lm_eval.utils import eval_logger

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _build_tasks_and_generate(
    tasks: Union[List[str], List[HarnessTask]], 
    lm: Union[lm_eval.api.model.LM, lm_eval.api.model.CachingLM], limit: int,
    write_out: bool
):
    task_dict = HarnessTaskManager.get_task_dict(tasks=tasks)
    task_dict, task_metadata = HarnessTaskManager.build_task_requests(
        task_dict=task_dict, lm=lm, limit=limit, write_out=write_out
    )
    requests = task_metadata["requests"]
    
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (task_metadata["padding_requests"][reqtype] > 0):
            for _ in range(task_metadata["padding_requests"][reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()
    
    return resps, task_dict, task_metadata


def harness_postprocessor(
    lm, 
    task_dict, 
    metadata, 
    config: EvaluatorConfig, 
    bootstrap_iters: int = 100000
):
    vals = collections.defaultdict(list)
    results, versions, configs, samples, requests, results_agg, groups_agg, _, task_hierarchy, num_fewshot = metadata.values()
    
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group, task = task
            if task is None:
                continue
        
        
        for key in task.instances[0].filtered_resps.keys():
            doc_iterator = (
                itertools.islice(
                    enumerate(task.test_docs()), lm.rank, config.limit, lm.world_size
                )
                if task.has_test_docs()
                else itertools.islice(
                    enumerate(task.validation_docs()), lm.rank, config.limit, lm.world_size
                )
            )
            
            for doc_id, doc in doc_iterator:
                # subset instances to only this document id ; sort by idx
                requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))
                requests.sort(key=lambda x: x.idx)
                metrics = task.process_results(
                    doc, [req.filtered_resps[key] for req in requests]
                )
                if config.log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[key] for req in requests],
                    }
                    example.update(metrics)
                    samples[task_name].append(example)
                for metric, value in metrics.items():
                    vals[(task_name, key, metric)].append(value)
                    
    if lm.world_size > 1:
        # if multigpu, then gather data across all ranks
        # first gather logged samples across all ranks
        for task_name, task_samples in list(samples.items()):
            full_samples = [None] * lm.world_size
            torch.distributed.all_gather_object(full_samples, task_samples)

            samples[task_name] = list(itertools.chain.from_iterable(full_samples))

        # then collect metrics across all ranks
        vals_torch = collections.defaultdict(list)
        for (task_name, key, metric), items in vals.items():
            numitem = 0
            if isinstance(items[0], tuple):
                numitem = len(items[0])

            if isinstance(items[0], (str, list, tuple)):
                # handle the string case
                gathered_items = [None] * lm.accelerator.num_processes
                torch.distributed.all_gather_object(gathered_items, items)

                gathered_item = list(itertools.chain.from_iterable(gathered_items))
            else:
                # distributed gather requires all ranks to have same dimensions
                # so we pad out with float32 min value
                pad_value = torch.finfo(torch.float32).min
                metrics_tensor = torch.tensor(items, device=lm.device)

                original_dtype = metrics_tensor.dtype  # store original dtype
                torch_device_tensor = lm.accelerator.pad_across_processes(
                    metrics_tensor.to(torch.float32), pad_index=pad_value
                )
                gathered_item = lm.accelerator.gather(torch_device_tensor)

                if numitem > 0:
                    gathered_filtered = gathered_item[gathered_item[:, 0] != pad_value]
                else:
                    gathered_filtered = gathered_item[gathered_item != pad_value]

                gathered_item = (
                    gathered_filtered.to(original_dtype).cpu().detach().numpy().tolist()
                )
                # reconvert if we were passed a tuple of values
                if numitem > 0:
                    gathered_item = [tuple(g) for g in gathered_item]

            if lm.rank == 0:
                vals_torch[(task_name, key, metric)] = gathered_item

        vals = vals_torch
    
    if lm.rank == 0:

        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for (task_name, key, metric), items in vals.items():
            task = task_dict[task_name]
            metric_key = metric + "," + key

            if isinstance(task, tuple):
                group_name, task = task
            else:
                group_name = None

            agg_fn = task.aggregation()[metric]
            results[task_name][metric_key] = agg_fn(items)
            results[task_name]["samples"] = len(items)

            # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
            # so we run them less iterations. still looking for a cleaner way to do this
            if bootstrap_iters > 0:
                stderr = lm_eval.api.metrics.stderr_for_metric(
                    metric=task.aggregation()[metric],
                    bootstrap_iters=min(bootstrap_iters, 100)
                    if metric in ["bleu", "chrf", "ter"]
                    else bootstrap_iters,
                )

                if stderr is not None and len(items) > 1:
                    results[task_name][metric + "_stderr" + "," + key] = stderr(items)
                else:
                    results[task_name][metric + "_stderr" + "," + key] = "N/A"

        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if task_list == []:
                    total_size = results[group]["samples"]
                else:
                    total_size = 0

                    for task in task_list:
                        metrics = results[task].copy()

                        if "alias" in metrics:
                            metrics.pop("alias")

                        current_size = metrics.pop("samples")
                        # TODO: There should be a way for users
                        #       to toggle between weighted and
                        #       unweighted averaging
                        # For unweighted averaging, use:
                        #     current_size = 1

                        all_stderr = []
                        for metric in [
                            key for key in metrics.keys() if "_stderr" not in key
                        ]:
                            stderr = "_stderr,".join(metric.split(","))
                            stderr_score = results[task][stderr]
                            if stderr_score == "N/A":
                                var_score = "N/A"
                            else:
                                var_score = stderr_score**2
                                all_stderr.append(stderr)

                            metric_score = results[task][metric]

                            if metric in results[group]:
                                results[group][metric] = (
                                    results[group][metric] * total_size
                                    + metric_score * current_size
                                ) / (total_size + current_size)
                                # $$s_z^2 = \frac{(n-1) s_x^2 + (m-1) s_y^2}{n+m-1} + \frac{nm(\bar x - \bar y)^2}{(n+m)(n+m-1)}.$$
                                if var_score == "N/A" or results[group][stderr] == "N/A":
                                    results[group][stderr] = "N/A"
                                else:
                                    results[group][stderr] = (
                                        (total_size - 1) * results[group][stderr]
                                        + (current_size - 1) * var_score
                                    ) / (
                                        total_size + current_size - 1
                                    ) + total_size * current_size / (
                                        (total_size + current_size)
                                        * (total_size + current_size - 1)
                                    ) * (
                                        results[group][metric] - metric_score
                                    ) ** 2
                            else:
                                results[group][metric] = metric_score
                                results[group][stderr] = var_score

                        total_size += current_size

                    for stderr in all_stderr:
                        results[group][stderr] = np.sqrt(results[group][stderr])

                results[group]["samples"] = total_size

        def print_tasks(task_hierarchy, results, tab=0):
            results_agg = collections.defaultdict(dict)
            groups_agg = collections.defaultdict(dict)

            (group_name, task_list), *_ = task_hierarchy.items()
            task_list = sorted(task_list)

            results_agg[group_name] = results[group_name].copy()
            # results_agg[group_name]["tab"] = tab
            if "samples" in results_agg[group_name]:
                results_agg[group_name].pop("samples")

            tab_string = " " * tab + "- " if tab > 0 else ""

            if "alias" in results_agg[group_name]:
                results_agg[group_name]["alias"] = (
                    tab_string + results_agg[group_name]["alias"]
                )
            else:
                results_agg[group_name]["alias"] = tab_string + group_name

            if len(task_list) > 0:
                groups_agg[group_name] = results[group_name].copy()
                # groups_agg[group_name]["tab"] = tab
                if "samples" in groups_agg[group_name]:
                    groups_agg[group_name].pop("samples")

                if "alias" in groups_agg[group_name]:
                    groups_agg[group_name]["alias"] = (
                        tab_string + groups_agg[group_name]["alias"]
                    )
                else:
                    groups_agg[group_name]["alias"] = tab_string + group_name

                for task_name in task_list:
                    if task_name in task_hierarchy:
                        _task_hierarchy = {
                            **{task_name: task_hierarchy[task_name]},
                            **task_hierarchy,
                        }
                    else:
                        _task_hierarchy = {
                            **{task_name: []},
                            **task_hierarchy,
                        }

                    _results_agg, _groups_agg = print_tasks(
                        _task_hierarchy, results, tab + 1
                    )
                    results_agg = {**results_agg, **_results_agg}
                    groups_agg = {**groups_agg, **_groups_agg}

            return results_agg, groups_agg

        results_agg = collections.defaultdict(dict)
        groups_agg = collections.defaultdict(dict)
        all_tasks_list = list(task_hierarchy.keys())
        left_tasks_list = []
        while True:
            add_tasks_list = list(k for k in results_agg.keys())
            left_tasks_list = sorted(list(set(all_tasks_list) - set(add_tasks_list)))
            if len(left_tasks_list) == 0:
                break

            _task_hierarchy = {
                k: v for k, v in task_hierarchy.items() if k in left_tasks_list
            }
            _results_agg, _groups_agg = print_tasks(_task_hierarchy, results)

            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

        for group_name, task_list in task_hierarchy.items():
            if task_list != []:
                num_fewshot[group_name] = num_fewshot[task_list[0]]

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }
        if config.log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict