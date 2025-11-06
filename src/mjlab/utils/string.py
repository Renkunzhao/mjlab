import re
from typing import Any, Dict, List, Pattern, Tuple


def resolve_expr(
  pattern_map: Dict[str, Any],
  names: List[str],
  default_val: Any = 0.0,
) -> List[Any]:
  # Pre-compile patterns in insertion order.
  compiled: List[Tuple[Pattern[str], Any]] = [
    (re.compile(pat), val) for pat, val in pattern_map.items()
  ]

  result: List[Any] = []
  for name in names:
    for pat, val in compiled:
      if pat.match(name):
        result.append(val)
        break
    else:
      result.append(default_val)
  return result


def filter_exp(exprs: List[str], names: List[str]) -> List[str]:
  patterns: List[Pattern] = [re.compile(expr) for expr in exprs]
  return [name for name in names if any(pat.match(name) for pat in patterns)]


def resolve_field(field: int | dict[str, int], names: list[str], default_val: Any = 0):
  return (
    resolve_expr(field, names, default_val)
    if isinstance(field, dict)
    else [field] * len(names)
  )


def resolve_param_to_list(
  param: float | dict[str, float], joint_names: list[str]
) -> list[float]:
  """Convert a parameter (float or dict) to a list matching joint order.

  Args:
    param: Single float or dict mapping joint names/regex patterns
      to values.
    joint_names: Ordered list of joint names.

  Returns:
    List of parameter values in the same order as joint_names.

  Raises:
    ValueError: If param is a dict and not all regex patterns match,
      or if multiple patterns match the same joint name.

  Example:
    >>> resolve_param_to_list(1.5, ["hip", "knee", "ankle"])
    [1.5, 1.5, 1.5]
    >>> resolve_param_to_list(
    ...   {"hip": 2.0, "knee": 1.5, "ankle": 1.0},
    ...   ["hip", "knee", "ankle"]
    ... )
    [2.0, 1.5, 1.0]
    >>> resolve_param_to_list(
    ...   {".*_hip": 2.0, ".*_knee": 1.5},
    ...   ["front_hip", "back_hip", "front_knee"]
    ... )
    [2.0, 2.0, 1.5]
  """
  if isinstance(param, dict):
    from mjlab.third_party.isaaclab.isaaclab.utils.string import (
      resolve_matching_names_values,
    )

    _, _, values = resolve_matching_names_values(
      param, joint_names, preserve_order=True
    )
    return values
  else:
    return [param] * len(joint_names)
