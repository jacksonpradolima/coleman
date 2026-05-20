"""
coleman.cli - Thin CLI wrapper over the library API.

Console-script entry-point ``coleman`` providing sub-commands:

* ``coleman run   --config run.yaml``
* ``coleman sweep --config base.yaml --grid key=v1,v2``
* ``coleman validate --config base.yaml``
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from coleman.api import load_spec, run, run_many, save_resolved, sweep
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import SweepAxis, SweepSpec


def _parse_kv(raw: str) -> tuple[str, list[Any]]:
    """Parse ``key=v1,v2,v3`` or ``key=range(start,stop)`` into (key, values).

    Parameters
    ----------
    raw : str
        A CLI key=value expression.

    Returns
    -------
    tuple[str, list[Any]]
        The parameter path and expanded values.

    Raises
    ------
    ValueError
        If the format is invalid.
    """
    if "=" not in raw:
        msg = f"Expected key=values format, got: {raw!r}"
        raise ValueError(msg)
    key, val_str = raw.split("=", 1)

    # range(start,stop[,step])
    if val_str.startswith("range(") and val_str.endswith(")"):
        inner = val_str[6:-1]
        parts = [int(p.strip()) for p in inner.split(",")]
        values: list[Any] = list(range(*parts))
    else:
        # Comma-separated values — try numeric first
        tokens = [t.strip() for t in val_str.split(",")]
        values = []
        for tok in tokens:
            try:
                values.append(int(tok))
            except ValueError:
                try:
                    values.append(float(tok))
                except ValueError:
                    values.append(tok)
    return key, values


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle ``coleman run``."""
    spec = load_spec(args.config, packs_dir=args.packs_dir)
    result = run(spec)
    print(f"run_id: {result.run_id}")  # noqa: T201
    if result.artifacts_dir:
        print(f"artifacts: {result.artifacts_dir}")  # noqa: T201


def _cmd_sweep(args: argparse.Namespace) -> None:
    """Handle ``coleman sweep``."""
    base = load_spec(args.config, packs_dir=args.packs_dir)

    params: dict[str, list[Any]] = {}
    for group in args.grid or []:
        for expr in group:
            key, vals = _parse_kv(expr)
            params[key] = vals

    sweep_spec = SweepSpec(
        axes=[SweepAxis(mode="grid", params=params)] if params else [],
    )
    specs = sweep(base, sweep_spec)
    print(f"Generated {len(specs)} specs")  # noqa: T201

    if args.dry_run:
        for s in specs:
            print(f"  run_id={compute_run_id(s)}")  # noqa: T201
        return

    workers = args.workers or base.execution.parallel_pool_size
    results = run_many(specs, max_workers=workers)
    for r in results:
        print(f"  run_id={r.run_id}")  # noqa: T201


def _cmd_validate(args: argparse.Namespace) -> None:
    """Handle ``coleman validate``."""
    try:
        spec = load_spec(args.config, packs_dir=args.packs_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"INVALID: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    rid = compute_run_id(spec)
    print(f"VALID  run_id={rid}")  # noqa: T201
    if args.resolve:
        out = save_resolved(spec, args.resolve)
        print(f"Resolved spec written to {out}")  # noqa: T201


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(prog="coleman", description="Coleman experiment runner")
    parser.add_argument(
        "--packs-dir", default=None, help="Root directory for config packs (default: <config_dir>/packs)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Execute a single run")
    p_run.add_argument("--config", required=True, help="Path to YAML config")

    # --- sweep ---
    p_sweep = sub.add_parser("sweep", help="Execute a parameter sweep")
    p_sweep.add_argument("--config", required=True, help="Path to base YAML config")
    p_sweep.add_argument("--grid", action="append", nargs="+", help="Grid params: key=v1,v2 or key=range(0,10)")
    p_sweep.add_argument("--dry-run", action="store_true", help="Print specs without executing")
    p_sweep.add_argument("--workers", type=int, default=None, help="Max parallel workers")

    # --- validate ---
    p_validate = sub.add_parser("validate", help="Validate a config file")
    p_validate.add_argument("--config", required=True, help="Path to YAML config")
    p_validate.add_argument("--resolve", default=None, help="Write resolved spec to this path")

    args = parser.parse_args(argv)

    dispatch = {
        "run": _cmd_run,
        "sweep": _cmd_sweep,
        "validate": _cmd_validate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
