"""Observe real shipped Hedge on authored CI loss panels, entirely locally.

Copy this file outside the checkout and run it with the noneditable v5 wheel.
The optional ``memory`` mode uses a real OTel SDK reader without an exporter,
network, global provider registration or export thread. This is not an efficacy
benchmark, a physical CI execution, or independent statistical observations.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.core.actions import Weight
from coleman.core.base import CoreContract
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig, ReferencePayload
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.state.contracts import AlgorithmState
from coleman.telemetry.export import LifecycleTelemetry, NoOpLifecycleTelemetry
from coleman.telemetry.lifecycle import LifecycleSnapshot, point

Mode = Literal["off", "noop", "memory"]
AT = datetime(2026, 1, 1, tzinfo=UTC)


def reference(name: str) -> Reference:
    """Return one consciously authored public version-one identity.

    Parameters
    ----------
    name : str
        Bounded public identifier. No source, plugin or registry is consulted.

    Returns
    -------
    Reference
        Exact supplied identity and semantic version1, without authentication.

    Raises
    ------
    ValueError
        Invalid identifier under the existing core reference contract.
    """
    return Reference(id=name, version="1")


class Workload(CoreContract):
    """Bound the example's actual learner count, experts and complete panels."""

    experts: Annotated[int, Field(strict=True, ge=2, le=32)] = 2
    panels: Annotated[int, Field(strict=True, ge=1, le=32)] = 2
    trajectories: Annotated[int, Field(strict=True, ge=1, le=4)] = 1


class Receipt(CoreContract):
    """Retain complete source-bound learner output, separate from resource cost."""

    workload: Workload
    mode: Mode
    final_states: Annotated[tuple[AlgorithmState[ReferencePayload], ...], Field(min_length=1, max_length=4)]
    snapshots: Annotated[tuple[LifecycleSnapshot, ...], Field(max_length=128)]
    sdk_metric_names: tuple[str, ...]

    @model_validator(mode="after")
    def _complete(self):
        """Keep all complete typed states and expected observations together."""
        if len(self.final_states) != self.workload.trajectories or any(
            state.payload.update_count != self.workload.panels or len(state.payload.weights) != self.workload.experts
            for state in self.final_states
        ):
            raise ValueError("A CI receipt requires every complete declared learner state")
        expected = 0 if self.mode == "off" else self.workload.panels * self.workload.trajectories
        if len(self.snapshots) != expected:
            raise ValueError("A CI receipt cannot omit declared operational snapshots")
        return self


def configuration(experts: int) -> ReferenceConfig:
    """Declare real constant Hedge over2..32 consciously authored CI experts.

    Parameters
    ----------
    experts : int
        Two through32 fixed experts with equal initial dimensionless weights.

    Returns
    -------
    ReferenceConfig
        Existing shipped constant-Hedge law, eta0.5, complete dimensionless
        bounded loss and explicitly authored transform/unknown lineage.

    Raises
    ------
    ValueError
        Unsupported expert count or numerical/semantic declaration. No learner
        is constructed, feedback fitted, sampling inferred or I/O performed.
    """
    declared = Workload(schema_version=1, experts=experts)
    members = tuple(reference(f"ci-rule-{index:02d}") for index in range(declared.experts))
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    loss = ScalarSpec(
        dimension=reference("authored-ci-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="authored-ci-bounded-loss-v1"),
    )
    return ReferenceConfig(
        schema_version=1,
        reference=reference("operational-ci-hedge"),
        lineage=lineage,
        profile="hedge-constant",
        eta=0.5,
        alpha=None,
        prior=InitialPrior(
            schema_version=1,
            reference=reference("operational-equal-ci-prior"),
            lineage=lineage,
            weights=tuple(Weight(member=member, value=1 / declared.experts) for member in members),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
        ),
        loss_spec=loss,
        transform=DefinitionBinding(
            reference=reference("authored-bounded-identity"),
            definition_fingerprint=fingerprint(
                {"profile": "bounded-identity-v1", "spec": loss.model_dump(mode="json"), "fitted_parameters": None}
            ),
        ),
    )


def _panel(learner: Hedge, index: int) -> FullPanel:
    """Create only the next explicitly authored synthetic loss panel."""
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=member,
                values=(
                    OutcomeValue(
                        spec=learner.loss_spec,
                        value=float((coordinate + index) % 2),
                        status="observed",
                        origin="synthetic",
                        transform=learner.transform.reference,
                    ),
                ),
            )
            for coordinate, member in enumerate(learner.members)
        )
    )


def _observer(mode: Mode):
    """Use an explicit local SDK provider only for the requested memory mode."""
    if mode != "memory":
        return NoOpLifecycleTelemetry(), None, None
    # Optional SDK imports are intentionally restricted to explicit memory mode.
    from opentelemetry.sdk.metrics import MeterProvider  # pylint: disable=import-outside-toplevel
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # pylint: disable=import-outside-toplevel

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=(reader,))
    return LifecycleTelemetry(provider.get_meter("coleman.operational.example")), reader, provider


def run(workload: Workload, mode: Mode = "noop") -> Receipt:
    """Run bounded real learners and observe only after each committed update.

    Parameters
    ----------
    workload : Workload
        At most4 trajectories,32 experts and32 panels. This is sequential own
        process work, not a queue, scheduler or replacement worker resource cap.
    mode : {'off', 'noop', 'memory'}, optional
        Off omits observation work, noop validates typed observations, memory
        additionally records actual SDK gauges. Default is noop.

    Returns
    -------
    Receipt
        Complete canonical learner state JSON and (when enabled) snapshots.
        Repeated snapshots do not become independent observations or fees.

    Raises
    ------
    ValueError, ImportError, RuntimeError
        Invalid input/mode, missing optional SDK or a typed learning refusal.

    Notes
    -----
    Owned provider is always shut down. No global provider, network, files,
    external effects or physical tests are created. Direct panels are actual
    operations; delivery deduplication belongs to existing replay/task routes.
    """
    workload = Workload.model_validate(workload)
    if mode not in {"off", "noop", "memory"}:
        raise ValueError("Unknown explicitly selected instrumentation mode")
    observer, reader, provider = _observer(mode)
    snapshots, states = [], []
    names: tuple[str, ...] = ()
    try:
        for trajectory in range(workload.trajectories):
            learner = Hedge(configuration(workload.experts), owner=reference(f"ci-owner-{trajectory}"))
            for index in range(workload.panels):
                result = learner.observe(_panel(learner, index))
                if result.status != "accepted":
                    raise RuntimeError("The bounded example reached a declared numerical refusal")
                if mode != "off":
                    state = learner.export_state()
                    observation = LifecycleSnapshot(
                        schema_version=1,
                        route="direct",
                        outcome="accepted",
                        observed_at=AT + timedelta(seconds=index + 1),
                        source_fingerprint=state.fingerprint,
                        algorithm=state.binding.implementation.reference,
                        state_codec=state.binding.state_codec,
                        points=(
                            point("updates", learner.update_count),
                            point("concentration", math.fsum(w.value * w.value for w in learner.weights)),
                        ),
                    )
                    observer.record(observation)
                    snapshots.append(observation)
            states.append(learner.export_state())
        if reader is not None:
            data = reader.get_metrics_data()
            if data is None:
                raise RuntimeError("The explicit memory reader returned no observed metrics")
            names = tuple(
                sorted(
                    metric.name
                    for resource in data.resource_metrics
                    for scope in resource.scope_metrics
                    for metric in scope.metrics
                )
            )
    finally:
        if provider is not None:
            provider.shutdown()
    return Receipt(
        schema_version=1,
        workload=workload,
        mode=mode,
        final_states=tuple(states),
        snapshots=tuple(snapshots),
        sdk_metric_names=names,
    )


def main() -> None:
    """Print one complete canonical receipt; no output means no completed report.

    Raises
    ------
    SystemExit
        Click handles help or invalid arguments before learner execution;
        a missing CLI extra exits with its explicit installation guidance.

    Notes
    -----
    Reads explicit command-line mode and writes stdout only after run returns.
    Click is imported only when this command is invoked; importing this module
    or calling ``run`` does not require the optional CLI extra. Unknown flags
    and help exit through Click; learner/provider errors propagate. This
    example owns no persistent checkpoint or physical execution.
    """
    try:
        click = import_module("click")
    except ImportError as error:
        raise SystemExit("Install the reviewed COLEMAN wheel with its [cli] extra") from error

    @click.option("--mode", type=click.Choice(("off", "noop", "memory")), default="noop", show_default=True)
    def _command(mode: Mode) -> None:
        """Run the explicitly selected local observation profile."""
        print(run(Workload(schema_version=1), mode).to_json())

    command = click.command(help=__doc__)(_command)
    command.main()


if __name__ == "__main__":
    main()
