"""Compare two authored CI observations with explicit units and subjective goals.

Run this standalone file with the wheel built from the v5 integration source.
The table is synthetic; this example executes objective operations, not a learner
or a production CI measurement. It writes only its JSON report to stdout.
"""

import json

from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.feasibility import (
    FiniteAlternative,
    FiniteFeasibilityRequest,
    ScalarConstraint,
    check_finite_feasibility,
    select_fallback,
)
from coleman.objectives.preferences import ObjectiveWeight, ObjectiveWeights, weight_objectives
from coleman.objectives.transforms import AffineLoss, apply_loss
from coleman.objectives.units import ConversionSpec, convert_quantity


def _reference(name: str) -> Reference:
    """Give each public synthetic dimension and operation an explicit version."""
    return Reference(id=name, version="1")


def _loss(name: str, horizon: Reference) -> QuantityMeaning:
    """Declare an actual normalized-loss dimension rather than a probability."""
    return QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference(name),
            semantics=UnitSemantics(unit="1", meaning=name + "-v1"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="normalized_loss",
        horizon=horizon,
    )


def main() -> None:
    """Run pure CI objective operations and print their deterministic report.

    Raises
    ------
    ValueError
        A required operation cannot produce an observed result.

    Notes
    -----
    Uses only authored synthetic values and declared modeling assumptions.
    Performs no fit, adaptive update, external execution, file write or clock
    read. Subjective weighted loss does not establish test effectiveness.
    """
    rows = (("unit", 0.8, 2000.0), ("integration", 1.0, 4000.0))
    horizon = _reference("synthetic-build")
    lineage = DefinitionLineage(
        source=_reference("authored-ci-table"),
        source_fingerprint=fingerprint([list(row) for row in rows]),
        declared_at=None,
        supersedes=None,
    )
    duration_ms = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("elapsed-ms"),
            semantics=UnitSemantics(unit="ms", meaning="elapsed-suite-wall-time-v1"),
            role="resource",
            minimum=0.0,
        ),
        interpretation="resource",
        horizon=horizon,
    )
    duration_s = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("elapsed-s"),
            semantics=UnitSemantics(unit="s", meaning="elapsed-suite-wall-time-v1"),
            role="resource",
            minimum=0.0,
        ),
        interpretation="resource",
        horizon=horizon,
    )
    credit = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("detected-fault-credit"),
            semantics=UnitSemantics(unit="1", meaning="synthetic-detected-fault-credit-v1"),
            role="objective",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="raw_outcome",
        horizon=horizon,
    )
    conversion = ConversionSpec(
        schema_version=1,
        reference=_reference("elapsed-ms-to-s"),
        lineage=lineage,
        input=duration_ms,
        output=duration_s,
    )
    duration_loss, detection_loss = _loss("duration-loss", horizon), _loss("detection-loss", horizon)
    elapsed = AffineLoss(
        schema_version=1,
        reference=_reference("elapsed-loss"),
        lineage=lineage,
        input=duration_s,
        output=duration_loss,
        lower=0.0,
        upper=4.0,
        direction="minimize",
        outside="reject",
        bound_basis="modeling_assumption",
    )
    detection = AffineLoss(
        schema_version=1,
        reference=_reference("detection-loss"),
        lineage=lineage,
        input=credit,
        output=detection_loss,
        lower=0.0,
        upper=1.0,
        direction="maximize",
        outside="reject",
        bound_basis="modeling_assumption",
    )
    preference = ObjectiveWeights(
        schema_version=1,
        reference=_reference("declared-ci-tradeoff"),
        lineage=lineage,
        terms=(
            ObjectiveWeight(quantity=detection_loss, weight=0.75),
            ObjectiveWeight(quantity=duration_loss, weight=0.25),
        ),
        output=ScalarSpec(
            dimension=_reference("subjective-ci-objective"),
            semantics=UnitSemantics(unit="1", meaning="declared-ci-objective-v1"),
            role="objective",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    objectives = []
    durations = []
    for name, detected, milliseconds in rows:
        converted = convert_quantity(conversion, Scalar(spec=duration_ms.spec, value=milliseconds, status="observed"))
        first = apply_loss(detection, Scalar(spec=credit.spec, value=detected, status="observed"))
        second = apply_loss(elapsed, converted.converted)
        if first.output is None or second.output is None:
            raise ValueError("The synthetic fixture must produce both observed losses")
        evaluated = weight_objectives(preference, (first.output, second.output))
        if evaluated.output is None or evaluated.output.value is None:
            raise ValueError("The synthetic fixture requires complete subjective objective support")
        objectives.append({"suite": name, "objective": round(evaluated.output.value, 6)})
        durations.append(converted.converted)
    members = tuple(_reference(row[0]) for row in rows)
    constraint = ScalarConstraint(
        schema_version=1,
        reference=_reference("elapsed-ceiling"),
        lineage=lineage,
        quantity=duration_s,
        lower=None,
        upper=3.0,
    )
    feasible = check_finite_feasibility(
        FiniteFeasibilityRequest(
            schema_version=1,
            universe=members,
            universe_complete=True,
            alternatives=tuple(
                FiniteAlternative(member=member, values=(duration,))
                for member, duration in zip(members, durations, strict=True)
            ),
            constraints=(constraint,),
        )
    )
    fallback = select_fallback(feasible, baseline=members[0])
    print(
        json.dumps(
            {
                "origin": "synthetic",
                "objectives": objectives,
                "duration_seconds": [value.value for value in durations],
                "feasible": [member.id for member in feasible.feasible_members],
                "fallback": fallback.kind,
                "selected": None if fallback.member is None else fallback.member.id,
            }
        )
    )


if __name__ == "__main__":
    main()
