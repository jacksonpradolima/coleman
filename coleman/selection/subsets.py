"""Expose the existing finite coverage solver as a genuine subset action."""

from collections.abc import Callable
from typing import Self

from pydantic import model_validator

from coleman.core.actions import AbstainAction, SubsetAction, SubsetSpace
from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import MAX_MEMBERS
from coleman.selection.contracts import CoverageProblem
from coleman.selection.resources import Method, SearchLimits
from coleman.selection.results import CoverageCheckpoint, CoverageResult
from coleman.selection.solver import solve_coverage


def coverage_space(problem: CoverageProblem) -> SubsetSpace:
    """Bind the complete declared solver scope to explicit versioned members.

    Parameters
    ----------
    problem : CoverageProblem
        Validated weighted union-coverage problem with additive modeled costs.

    Returns
    -------
    SubsetSpace
        Actual search scope, including its content-bound semantic version.

    Raises
    ------
    ValueError
        The supplied scope exceeds the action contract's 4096-member ceiling.
        Nothing is truncated and no smaller benchmark claim is substituted.

    Notes
    -----
    Mandatory identities and resource feasibility remain in the full retained
    solver result; the action shape alone is not a feasibility certificate.
    """
    problem = CoverageProblem.model_validate(problem)
    candidates = problem.scoped_candidates()
    if len(candidates) > MAX_MEMBERS:
        raise ValueError("Coverage action scope exceeds 4096; declare an explicit restricted problem")
    version = problem.fingerprint
    maximum = problem.constraints.maximum_selected
    return SubsetSpace(
        feasible_set=Reference(id="coverage-scope", version=version),
        members=tuple(Reference(id=row.candidate_id, version=version) for row in candidates),
        minimum_selected=0,
        maximum_selected=len(candidates) if maximum is None else min(maximum, len(candidates)),
    )


class CoverageSelection(CoreContract):
    """Retain an actual subset or abstention together with the original proof.

    Attributes
    ----------
    result : CoverageResult
        Unmodified objective, units, scope, feasibility, bounds and search status.
    action : SubsetAction or AbstainAction
        A chosen unordered subset for a feasible incumbent, including empty;
        otherwise explicit abstention. No permutation, reward, joint inclusion
        probability or combinatorial-bandit assumption is manufactured.

    Notes
    -----
    Version1 is a new adapter, not a migration of old coverage artifacts. Reading
    this envelope does not execute the solver or publish an artifact.
    """

    result: CoverageResult
    action: SubsetAction | AbstainAction

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Reject substituted subsets, proof scopes and abstention reasons."""
        expected = _action(self.result)
        if self.action != expected:
            raise ValueError("Coverage action differs from its retained solver result")
        return self


def _action(result: CoverageResult) -> SubsetAction | AbstainAction:
    """Map one already validated solver outcome without interpreting null as zero."""
    space = coverage_space(result.problem)
    if not result.feasible:
        return AbstainAction(schema_version=1, space=space, reason=result.termination_reason)
    chosen = set(result.selected_ids or ())
    return SubsetAction(schema_version=1, space=space, selected=tuple(row for row in space.members if row.id in chosen))


def select_coverage_subset(
    problem: CoverageProblem,
    *,
    method: Method = "greedy_density_v1",
    limits: SearchLimits | None = None,
    checkpoint: CoverageCheckpoint | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> CoverageSelection:
    """Solve with the existing qualified coverage core and expose its subset.

    Parameters
    ----------
    problem : CoverageProblem
        Complete supplied universe, optionally explicitly restricted.
    method : str, optional
        Existing solver profile, default ``greedy_density_v1``. Exact-search
        claims apply only to the retained solver's qualified optimal status.
    limits : SearchLimits or None, optional
        Existing bounded computation limits; defaults remain unchanged.
    checkpoint : CoverageCheckpoint or None, optional
        Existing verified incumbent hint. Search restarts rather than pretending
        that unretained proof/frontier state continues.
    cancelled : callable or None, optional
        Existing cooperative cancellation callback. Any feasible interrupted
        incumbent remains an actual subset with its honest retained status.

    Returns
    -------
    CoverageSelection
        Actual action plus unmodified typed proof, not a ranked candidate list.

    Raises
    ------
    ValueError
        Invalid problem, unsupported method or an oversized action scope.

    Notes
    -----
    Performs bounded pure computation, without acquisition, storage or sampling.
    Shared/nonadditive physical costs are outside the coverage formulation.
    """
    coverage_space(problem)
    result = solve_coverage(problem, method=method, limits=limits, checkpoint=checkpoint, cancelled=cancelled)
    return CoverageSelection(schema_version=1, result=result, action=_action(result))
