from typing import Optional
from pydantic import BaseModel, Field


class EvaluatorConfig(BaseModel):
    num_fewshot: Optional[int] = Field(default=None)
    batch_size: Optional[int] = Field(
        default=1, description="The batch size for evaluation."
    )
    max_batch_size: Optional[int] = Field(
        default=None, description="Maximal batch size used."
    )

    device: Optional[str] = Field(
        default=None, description="Device to compute on (e.g. cuda:0, cpu)."
    )
    use_cache: Optional[str] = Field(
        default=None, description="Path to load evaluations from cache."
    )
    limit: Optional[float] = Field(
        default=None, description="Limit for number of examples."
    )
    decontamination_ngrams_path: Optional[str] = (
        None  # To be removed by the harness as it's unused.
    )
    output_path: Optional[str] = Field(
        default=None, description="Path to store the logs."
    )
    check_integrity: bool = Field(
        default=False, description="Check integrity for tasks."
    )
    write_out: bool = Field(
        default=False, description="Print prompt for the first few documents."
    )
    log_samples: bool = Field(
        default=True,
        description="Write all model outputs and documents for per-sample measurement and analysis.",
    )
    show_config: bool = Field(
        default=True,
        description="Show full config of tasks at the end of evaluation.",
    )
    include_path: Optional[str] = Field(
        None, description="Additional path for external tasks."
    )
    gen_kwargs: Optional[str] = Field(
        None,
        description="String arguments for model generation on certain tasks.",
    )
    verbosity: str = Field(
        "INFO", description="Log error when tasks are not registered."
    )
