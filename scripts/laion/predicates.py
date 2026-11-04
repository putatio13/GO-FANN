#!/usr/bin/env python3
"""Shared LAION range-predicate evaluation helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


RANGE_COLUMN_INDEX = {
    "width": 0,
    "original_width": 0,
    "height": 1,
    "original_height": 1,
    "similarity": 2,
}


def range_column(term: dict[str, Any]) -> int:
    if "column" in term:
        return int(term["column"])
    field = str(term.get("field") or "")
    if field not in RANGE_COLUMN_INDEX:
        raise ValueError(f"unsupported range field {field!r}")
    return RANGE_COLUMN_INDEX[field]


def term_bounds(term: dict[str, Any]) -> tuple[float | None, float | None]:
    lower = term.get("lo", term.get("min", term.get("lower")))
    upper = term.get("hi", term.get("max", term.get("upper")))
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def range_term_mask(values: np.ndarray, term: dict[str, Any]) -> np.ndarray:
    column_values = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(column_values)
    operator = str(term.get("op") or "between").lower()
    lower, upper = term_bounds(term)

    if operator == "gt":
        if lower is None:
            raise ValueError("gt range predicate requires a lower bound")
        mask &= column_values > np.float32(lower)
        return mask
    if operator == "lt":
        if upper is None:
            raise ValueError("lt range predicate requires an upper bound")
        mask &= column_values < np.float32(upper)
        return mask
    if operator != "between":
        raise ValueError(f"unsupported range predicate op {operator!r}")

    if lower is not None:
        if bool(term.get("inclusive_min", True)):
            mask &= column_values >= np.float32(lower)
        else:
            mask &= column_values > np.float32(lower)
    if upper is not None:
        if bool(term.get("inclusive_max", True)):
            mask &= column_values <= np.float32(upper)
        else:
            mask &= column_values < np.float32(upper)
    if lower is None and upper is None:
        raise ValueError("between range predicate requires at least one bound")
    return mask


def range_predicate_mask(
    range_values: np.ndarray,
    predicates: Sequence[dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(range_values)
    if values.ndim != 2:
        raise ValueError(f"range values must be a matrix, got shape {values.shape}")
    if not predicates:
        raise ValueError("range predicate has no terms")

    mask = np.ones((values.shape[0],), dtype=bool)
    for term in predicates:
        column = range_column(term)
        if column < 0 or column >= values.shape[1]:
            raise ValueError(
                f"range column {column} is outside matrix width {values.shape[1]}"
            )
        mask &= range_term_mask(values[:, column], term)
        if not mask.any():
            break
    return mask
