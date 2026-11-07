"""Refuse malformed public proposals before exact numerical work."""

from typing import cast

import pytest

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.objectives.controls import ProjectionOutput
from coleman.optimization.mixture import solve_mixture_output
from tests.optimization.test_geometry_validation import bounds, origin


@pytest.mark.parametrize("form", ["empty", "mutable", "over_capacity"])
def test_public_solver_rejects_unbounded_or_mutable_proposal_inventory(form):
    current = origin((0.5, 0.5))
    proposed = current.weights
    if form == "empty":
        proposed = ()
    elif form == "mutable":
        proposed = cast(tuple[Weight, ...], list(proposed))
    else:
        proposed = tuple(
            Weight(member=Reference(id=f"expert-{index:04}", version="1"), value=1 / 4097) for index in range(4097)
        )
    declared, output = bounds(2), ProjectionOutput()
    before = current.to_json(), declared.to_json()
    with pytest.raises(ValueError, match="tuple of one through 4096 weights"):
        solve_mixture_output(proposal=proposed, bounds=declared, origin=current, output=output)
    assert (current.to_json(), declared.to_json()) == before


@pytest.mark.parametrize("mismatch", ["member", "version", "missing_member"])
def test_public_solver_requires_complete_exact_proposal_member_versions(mismatch):
    current = origin((0.5, 0.5))
    if mismatch == "missing_member":
        proposed = (Weight(member=current.weights[0].member, value=1.0),)
    else:
        member = Reference(
            id="another-expert" if mismatch == "member" else current.weights[0].member.id,
            version="2" if mismatch == "version" else "1",
        )
        proposed = (Weight(member=member, value=0.5), current.weights[1])
    declared, output = bounds(2), ProjectionOutput()
    before = tuple(weight.model_dump_json() for weight in proposed), current.to_json(), declared.to_json()
    with pytest.raises(ValueError, match="every exact geometry expert and version"):
        solve_mixture_output(proposal=proposed, bounds=declared, origin=current, output=output)
    assert (tuple(weight.model_dump_json() for weight in proposed), current.to_json(), declared.to_json()) == before
