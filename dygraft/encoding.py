"""Runtime helpers extracted from the experiment implementation."""
from __future__ import annotations

import random


import numpy as np


import pandas as pd


import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode(dataset, encoded: torch.Tensor, info: dict) -> pd.DataFrame:
    d_num = dataset.d_numerical
    array = encoded.detach().cpu().numpy()
    num_all = dataset.int_inverse(dataset.num_inverse(array[:, :d_num])).astype(np.float32)
    cat_all = dataset.cat_inverse(array[:, d_num:])
    n_target = len(info["target_col_idx"])
    # Match TabDiff's released split_num_cat_target convention.  Regression
    # targets are prepended to the numerical block; classification targets are
    # prepended to the categorical block.  The second regression branch keeps
    # old mechanism checkpoints (which encoded the target categorically)
    # readable, but new official-protocol runs use the first branch.
    regression_target_is_numerical = (
        info["task_type"] == "regression"
        and d_num == len(info["num_col_idx"]) + n_target
    )
    if regression_target_is_numerical:
        target, num = num_all[:, :n_target], num_all[:, n_target:]
        cat = cat_all
    else:
        num = num_all
        target, cat = cat_all[:, :n_target], cat_all[:, n_target:]
    mapping = {int(key): int(value) for key, value in info["idx_mapping"].items()}
    out = pd.DataFrame(index=np.arange(len(array)))
    for original in range(len(info["column_names"])):
        name = info["column_names"][original]
        mapped = mapping[original]
        if original in info["num_col_idx"]:
            out[name] = num[:, mapped]
        elif original in info["cat_col_idx"]:
            out[name] = cat[:, mapped - len(info["num_col_idx"])]
        else:
            out[name] = target[:, mapped - len(info["num_col_idx"]) - len(info["cat_col_idx"])]
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].astype(str).str.strip()
    return out
