"""Run three pure reference learners on one public constructed full-loss table.

This example imports no replay engine, registry, local store, runner or provider.
It compares numerical rules and complete continuation, without an efficacy claim.
Run the file with the base installation from any working directory.
"""

from __future__ import annotations

import json
from math import log

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig, ReferencePayload, ReferenceProfile
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.state.codec import read_algorithm_state

LOSS_TABLE = ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.25, 0.75, 0.5))


def reference(name: str) -> Reference:
    """Declare a local exact identity without discovering or registering code.

    Parameters
    ----------
    name : str
        Bounded logical identifier, supplied explicitly by this example.

    Returns
    -------
    Reference
        Immutable version-one logical identity.
    """
    return Reference(id=name, version="1")


def configuration(profile: ReferenceProfile) -> ReferenceConfig:
    """Bind a positive prior and one precise bounded-loss mathematical rule.

    Parameters
    ----------
    profile : str
        Constant Hedge, cumulative-current-rate anytime Hedge, or excluding-
        origin Fixed Share. Every member has an observed scalar loss each round.

    Returns
    -------
    ReferenceConfig
        Complete fixed configuration with explicit source/selection unknowns.
    """
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    members = tuple(reference("expert-" + name) for name in ("a", "b", "c"))
    prior = InitialPrior(
        schema_version=1,
        reference=reference("constructed-positive-prior"),
        lineage=lineage,
        weights=tuple(
            Weight(member=member, value=value) for member, value in zip(members, (0.5, 0.25, 0.25), strict=True)
        ),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    loss = ScalarSpec(
        dimension=reference("constructed-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="constructed-full-information-loss-v1"),
    )
    transform = DefinitionBinding(
        reference=reference("already-bounded-identity"),
        definition_fingerprint=fingerprint({"operation": "already-bounded-identity-v1"}),
    )
    return ReferenceConfig(
        schema_version=1,
        reference=reference("constructed-" + profile),
        lineage=lineage,
        profile=profile,
        prior=prior,
        loss_spec=loss,
        transform=transform,
        eta=None if profile == "hedge-anytime-cumulative" else log(2.0),
        alpha=0.1 if profile == "fixed-share-excluding-origin" else None,
        anytime_multiplier=2.0 if profile == "hedge-anytime-cumulative" else None,
    )


def panel(config: ReferenceConfig, losses: tuple[float, ...]) -> FullPanel:
    """Construct one exact complete observed panel in the configured loss units.

    Parameters
    ----------
    config : ReferenceConfig
        Fixed exact member versions, scalar loss interpretation and transform.
    losses : tuple of float
        One finite value in [0,1] per configured member, in canonical order.

    Returns
    -------
    FullPanel
        Explicit synthetic observed losses; an unavailable value is never zero.

    Raises
    ------
    ValueError, TypeError
        Wrong cardinality or an invalid typed scalar observation.
    """
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=value,
                        status="observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for weight, value in zip(config.prior.weights, losses, strict=True)
        )
    )


def run_profile(learner: Hedge | AnytimeHedge | FixedShare) -> dict[str, object]:
    """Apply the actual table and compare continuation through a bounded codec.

    Parameters
    ----------
    learner : Hedge or AnytimeHedge or FixedShare
        Fresh independently owned shipped reference on this example's catalogue.

    Returns
    -------
    dict of str to object
        Profile, actual implementation identity, three U-clocked emitted vectors,
        complete state fingerprint and byte-identical restored-continuation flag.

    Raises
    ------
    ValueError
        A numerical failure or a different restored continuation is observed.
    """
    trace = []
    for losses in LOSS_TABLE[:2]:
        result = learner.observe(panel(learner.config, losses))
        if result.status != "accepted":
            raise ValueError("Constructed reference update failed numerically")
        trace.append({"U": result.update_count, "weights": [item.value for item in result.weights]})
    snapshot = learner.export_state()
    checked = read_algorithm_state(
        snapshot.to_json(),
        payload_type=ReferencePayload,
        expected=learner.binding,
        maximum_bytes=learner.config.limits.maximum_bytes,
    )
    restored = type(learner)(learner.config, owner=learner.binding.owner)
    restored.restore_state(checked)
    last = panel(learner.config, LOSS_TABLE[2])
    result, continued = learner.observe(last), restored.observe(last)
    if result.status != "accepted" or continued.status != "accepted":
        raise ValueError("Constructed continuation failed numerically")
    equal = learner.export_state().to_json() == restored.export_state().to_json()
    if not equal:
        raise ValueError("Complete restored reference continuation differs")
    trace.append({"U": result.update_count, "weights": [item.value for item in result.weights]})
    return {
        "profile": learner.config.profile,
        "implementation": learner.binding.implementation.reference.model_dump(mode="json"),
        "trace": trace,
        "state_fingerprint": learner.export_state().fingerprint,
        "restored_continuation_equal": equal,
    }


def main() -> None:
    """Print deterministic finite reference traces without executing a domain.

    Returns
    -------
    None
        One JSON object is written to standard output; no file is persisted.
    """
    learners = (
        Hedge(configuration("hedge-constant"), owner=reference("constant-owner")),
        AnytimeHedge(configuration("hedge-anytime-cumulative"), owner=reference("anytime-owner")),
        FixedShare(configuration("fixed-share-excluding-origin"), owner=reference("share-owner")),
    )
    print(
        json.dumps(
            {"schema_version": 1, "loss_table": LOSS_TABLE, "profiles": [run_profile(item) for item in learners]},
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
