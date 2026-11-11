"""Runtime helpers extracted from the experiment implementation."""
from __future__ import annotations

import numpy as np


import pandas as pd


def canonical_categorical(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    missing = result.isna() | result.str.lower().isin({"", "?", "na", "n/a", "nan", "unknown"})
    return result.mask(missing, "<MISSING_LIKE>")


def predicate_mask(frame: pd.DataFrame, predicate: dict) -> np.ndarray:
    left_column, operator = predicate["left_column"], predicate["operator"]
    if left_column not in frame:
        raise KeyError(left_column)
    left = frame[left_column]
    if predicate["right_kind"] == "column":
        right_column = predicate["right_column"]
        if right_column not in frame:
            raise KeyError(right_column)
        right = frame[right_column]
    else:
        right = predicate["right_value"]
    tolerance = float(predicate.get("tolerance", 0))
    if operator in {"==", "!="}:
        if pd.api.types.is_numeric_dtype(left):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
            if isinstance(right, pd.Series):
                right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
            else:
                right_values = float(right)
            result = np.isclose(left_values, right_values, rtol=0, atol=max(tolerance, 1e-7), equal_nan=False)
        else:
            left_values = canonical_categorical(left)
            right_values = canonical_categorical(right) if isinstance(right, pd.Series) else str(right).strip()
            result = left_values.eq(right_values).fillna(False).to_numpy()
        return ~result if operator == "!=" else result
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce") if isinstance(right, pd.Series) else float(right)
    if operator == "<=": result = left_values.le(right_values + tolerance)
    elif operator == "<": result = left_values.lt(right_values + tolerance)
    elif operator == ">=": result = left_values.ge(right_values - tolerance)
    else: result = left_values.gt(right_values - tolerance)
    return result.fillna(False).to_numpy()
