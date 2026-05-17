"""
Sweep engine — grid, zip, and repeat expansion.

Given a base :class:`RunSpec` and a :class:`SweepSpec`, the engine
produces a deterministically-ordered sequence of concrete ``RunSpec``
instances with every parameter combination materialised.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from coleman.spec.models import RunSpec


class SweepAxis(BaseModel):
    """A single sweep dimension.

    Parameters
    ----------
    mode : str
        ``"grid"`` for Cartesian product, ``"zip"`` for element-wise
        pairing.
    params : dict[str, list[Any]]
        Mapping of dotted parameter paths to the list of values to sweep.
    """

    mode: Literal["grid", "zip"] = "grid"
    params: dict[str, list[Any]] = Field(default_factory=dict)


class SweepSpec(BaseModel):
    """Multi-axis sweep definition.

    Parameters
    ----------
    axes : list[SweepAxis]
        Sweep dimensions.  Grid axes are combined via Cartesian product;
        zip axes are combined element-wise.
    seeds : list[int] | None
        If set, each generated spec is further replicated once per seed
        (the seed is written to ``execution.seed``).
    """

    axes: list[SweepAxis] = Field(default_factory=list)
    seeds: list[int] | None = None


def _set_nested(data: object, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path.

    Parameters
    ----------
    data : object
        Root object to mutate. Must be a dictionary-like mapping at runtime.
    dotted_key : str
        Dot-separated path (e.g. ``"algorithm.ucb.timerank.c"``).
    value : Any
        Value to assign.

    Raises
    ------
    ValueError
        If an intermediate path component is not a mapping.
    """
    parts = dotted_key.split(".")
    current: object = data
    for idx, part in enumerate(parts[:-1]):
        if not isinstance(current, dict):
            path_so_far = ".".join(parts[:idx])
            msg = (
                f"Cannot set nested key {dotted_key!r}: "
                f"path component {path_so_far!r} is not a mapping "
                f"(found {type(current).__name__})."
            )
            raise ValueError(msg)

        current_dict = cast(dict[str, Any], current)

        if part in current_dict:
            next_obj = current_dict[part]
            if not isinstance(next_obj, dict):
                path_so_far = ".".join(parts[: idx + 1])
                msg = (
                    f"Cannot set nested key {dotted_key!r}: "
                    f"path component {path_so_far!r} is not a mapping "
                    f"(found {type(next_obj).__name__})."
                )
                raise ValueError(msg)
            current = next_obj
        else:
            new_container: dict[str, Any] = {}
            current_dict[part] = new_container
            current = new_container

    if not isinstance(current, dict):
        path_so_far = ".".join(parts[:-1])
        msg = (
            f"Cannot set nested key {dotted_key!r}: "
            f"path component {path_so_far!r} is not a mapping "
            f"(found {type(current).__name__})."
        )
        raise ValueError(msg)

    current_dict = cast(dict[str, Any], current)
    current_dict[parts[-1]] = value


def _expand_axis(axis: SweepAxis) -> list[dict[str, Any]]:
    """Expand one axis into a list of override dicts.

    Parameters
    ----------
    axis : SweepAxis
        A single sweep axis definition.

    Returns
    -------
    list[dict[str, Any]]
        Each element is a flat ``{dotted_key: value}`` mapping.
    """
    sorted_keys = sorted(axis.params.keys())

    if axis.mode == "grid":
        sorted_values = [axis.params[k] for k in sorted_keys]
        combos = list(itertools.product(*sorted_values))
    elif axis.mode == "zip":
        sorted_values = [axis.params[k] for k in sorted_keys]
        lengths = {len(v) for v in sorted_values}
        if len(lengths) > 1:
            msg = (
                "All parameter lists for a 'zip' sweep axis must have the same "
                f"length; got lengths {lengths} for keys {sorted_keys}."
            )
            raise ValueError(msg)
        combos = list(zip(*sorted_values, strict=True))
    else:
        msg = f"Unknown sweep mode: {axis.mode!r}"
        raise ValueError(msg)

    return [dict(zip(sorted_keys, combo, strict=True)) for combo in combos]


def expand_sweep(base: RunSpec, sweep: SweepSpec) -> list[RunSpec]:
    """Expand *base* × *sweep* into a deterministic list of ``RunSpec``.

    Parameters
    ----------
    base : RunSpec
        Template spec to override.
    sweep : SweepSpec
        Sweep definition (axes + optional seeds).

    Returns
    -------
    list[RunSpec]
        Concrete specs in deterministic order.
    """
    if not sweep.axes:
        return [base.model_copy(deep=True)]

    # Expand each axis independently.
    axis_expansions = [_expand_axis(ax) for ax in sweep.axes]

    # Cartesian product across axes.
    combined = list(itertools.product(*axis_expansions))

    specs: list[RunSpec] = []
    for combo in combined:
        data = base.model_dump()
        for overrides in combo:
            for dotted_key, value in sorted(overrides.items()):
                _set_nested(data, dotted_key, value)
        specs.append(RunSpec.model_validate(data))

    # Seed replication.
    if sweep.seeds is not None:
        seeded: list[RunSpec] = []
        for spec in specs:
            for seed in sweep.seeds:
                d = spec.model_dump()
                _set_nested(d, "execution.seed", seed)
                seeded.append(RunSpec.model_validate(d))
        specs = seeded

    return specs
