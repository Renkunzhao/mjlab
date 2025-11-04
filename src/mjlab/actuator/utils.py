"""Utility functions for actuator parameter resolution."""

from __future__ import annotations

import torch

from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)


def resolve_param(
  param: float | dict[str, float],
  target_name: str,
  default: float | None = None,
) -> float:
  """Resolve a parameter value for a single target joint.

  Used during spec editing when configuring MuJoCo actuator elements.

  Args:
    param: Parameter value (scalar or dict mapping patterns to values).
    target_name: Name of the target joint.
    default: Default value if target doesn't match any pattern. If None, raises error.

  Returns:
    The resolved parameter value for this target.

  Raises:
    ValueError: If param is a dict and target doesn't match any pattern (and no default).
  """
  if isinstance(param, dict):
    base_name = target_name.split("/")[-1]
    indices, _, values = resolve_matching_names_values(param, [base_name])
    if not indices:
      if default is not None:
        return default
      raise ValueError(
        f"Target '{base_name}' does not match any pattern in parameter dict. "
        f"Available patterns: {list(param.keys())}"
      )
    return values[0]
  return float(param)


def resolve_param_tensor(
  param: float | dict[str, float],
  target_names: list[str],
  default: float,
  device: str,
) -> torch.Tensor:
  """Resolve parameter values for all target joints as a tensor.

  Used during initialization when setting up runtime tensors.

  Args:
    param: Parameter value (scalar or dict mapping patterns to values).
    target_names: List of target joint names.
    default: Default value for unmatched targets.
    device: Device for the output tensor.

  Returns:
    Tensor of shape [num_targets] with parameter values.
  """
  values = torch.full(
    (len(target_names),),
    default,
    dtype=torch.float32,
    device=device,
  )

  if isinstance(param, (int, float)):
    values[:] = float(param)
  elif isinstance(param, dict):
    indices, _, matched_values = resolve_matching_names_values(param, target_names)
    values[indices] = torch.tensor(matched_values, dtype=torch.float32, device=device)
  else:
    raise TypeError(f"Parameter must be float or dict, got {type(param)}")

  return values
