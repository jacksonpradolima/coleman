"""Chronological synthetic feature/predict/eligible-label flow with real recovery.

Run with the noneditable installed wheel. This authored input is a mechanics
example, not independent histories or evidence of forecasting performance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path

from coleman.core.decisions import DecisionRequest
from coleman.core.feedback import AggregateFeedback, FeedbackBatch, OutcomeValue
from coleman.core.numeric import Scalar
from coleman.core.primitives import fingerprint
from coleman.core.splits import TemporalObservation
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.objectives.bindings import DefinitionBinding
from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.replay.contracts import ReplayFeedback, ReplayOpportunity, ReplayPolicy, ReplayTarget, ValueOriginEvidence
from coleman.replay.retained import RetainedReplay, ordinal_decision_id
from coleman.replay.retained_contracts import OrdinalRetention, RetainedContinuation
from coleman.results.domain import Measurement
from coleman.state.components import ComponentBinding, ComponentCodec, child_frontier
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility


def load_model():
    """Load only the explicitly supplied public sibling host source.

    Returns
    -------
    module
        Source-bound model module; no plugin discovery or arbitrary imports.
    """
    name = "authored_continuous_forecast_model"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name("model.py"))
        if specification is None or specification.loader is None:
            raise RuntimeError("The explicitly copied model.py is required")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


MODEL = load_model()
NOW = datetime(2026, 1, 1, tzinfo=UTC)
EPOCH = MODEL.reference("constructed-forecast-source-epoch")
FEATURES = ((1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (0.0, 1.0))
LABELS = (2.0, 0.0, -1.0, 1.0)


def implementation() -> DefinitionBinding:
    """Bind the actual public feature/label generator and chronological driver.

    Returns
    -------
    DefinitionBinding
        SHA-256 of this explicitly selected source, without authentication or
        a claim that arbitrary callers supplied these exact authored labels.
    """
    return DefinitionBinding(
        reference=MODEL.reference("constructed-forecast-flow"),
        definition_fingerprint="sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def instant(offset: float):
    """Convert an authored minute offset to an explicit UTC clock.

    Parameters
    ----------
    offset : float
        Minute coordinate relative to the declared example origin.

    Returns
    -------
    datetime
        Zoned UTC instant, independent from wall-clock execution.
    """
    return NOW + timedelta(minutes=offset)


def history() -> ObservationHistory:
    """Declare public synthetic feature revisions with genuine as-of selection.

    Returns
    -------
    ObservationHistory
        Eight independent-coordinate facts, not eight independent observations.
        Future rows remain outside earlier selected-view fingerprints.
    """
    config = MODEL.configuration()
    rows = []
    for index, values in enumerate(FEATURES):
        for spec, value in zip(config.features, values, strict=True):
            rows.append(
                ObservationRevision(
                    schema_version=1,
                    metadata=TemporalObservation(
                        observation_id=f"step-{index}-{spec.dimension.id}",
                        group_id="constructed-chronological-path",
                        event_at=instant(2 * index),
                        available_at=instant(2 * index),
                    ),
                    entity_id=spec.dimension.id,
                    received_at=instant(2 * index),
                    supersedes=None,
                    measurement=Measurement(
                        name=spec.dimension.id,
                        value=value,
                        unit=spec.semantics.unit,
                        meaning=spec.semantics.meaning,
                        direction="none",
                        status="observed",
                    ),
                )
            )
    return ObservationHistory(
        schema_version=1,
        source_id=MODEL.SOURCE.id,
        source_version=MODEL.SOURCE.version,
        revisions=tuple(rows),
    )


def opportunity(host, replay, index: int) -> ReplayOpportunity:
    """Derive actual pre-decision features from the replay's selected source view.

    Parameters
    ----------
    host : ForecastHost
        Actual independently owned predictor application.
    replay : RetainedReplay
        Existing admitted source and delivery owner.
    index : int
        Source-owned opportunity ordinal, zero through3.

    Returns
    -------
    ReplayOpportunity
        Original immutable feature vector and one-step future label horizon.

    Raises
    ------
    ValueError
        A required input is absent, unobserved or has incompatible units.
    """
    at = instant(2 * index)
    snapshot = replay.eligible_snapshot(at)
    by_id = {row.metadata.observation_id: row for row in snapshot.observations}
    values = []
    for spec in host.config.features:
        row = by_id.get(f"step-{index}-{spec.dimension.id}")
        if row is None or row.measurement.status != "observed":
            raise ValueError("Required point-in-time feature is unavailable")
        if (row.measurement.unit, row.measurement.meaning) != (spec.semantics.unit, spec.semantics.meaning):
            raise ValueError("Point-in-time feature units or meaning differ")
        values.append(Scalar(spec=spec, value=row.measurement.value, status="observed"))
    return ReplayOpportunity(
        schema_version=1,
        request=DecisionRequest(
            schema_version=1,
            task=host.task,
            decision_id=ordinal_decision_id(EPOCH, index),
            round=index,
            space=host.space,
            objective=MODEL.OBJECTIVE,
            inputs=tuple(values),
            timing=DecisionTiming(
                decision_at=at,
                information_cutoff=at,
                target_start=at,
                target_end=at + timedelta(minutes=1),
                mode="future",
                eligible_input_fingerprint=snapshot.fingerprint,
            ),
        ),
        context=None,
        target=ReplayTarget(
            reference=MODEL.reference(f"constructed-target-{index}"),
            horizon=MODEL.HORIZON,
            groups=(MODEL.reference("single-constructed-path"),),
        ),
    )


def feedback(
    record,
    value: float,
    *,
    revision: tuple[str, str | None] = ("initial", None),
    receipt: float | None = None,
    delay: int = 0,
) -> ReplayFeedback:
    """Author one synthetic signed label with separate maturity and receipt clocks.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Exact original score execution and one-step target.
    value : float
        Signed dimensionless response, including valid zero.
    revision : tuple of str and str or None, optional
        Explicit revision identity and predecessor, default('initial',None).
        A late correction is audit-only after application.
    receipt : float or None, optional
        First receipt in authored minutes; default target-end plus half a minute.
    delay : int, optional
        Intervening emitted opportunities at actual application; default0.

    Returns
    -------
    ReplayFeedback
        Whole-action supervised observation, never an expert-loss panel.
    """
    timing = record.opportunity.request.timing
    if timing is None:
        raise ValueError("The public example requires original explicit clocks")
    mature = timing.target_end
    received = mature + timedelta(seconds=30) if receipt is None else instant(receipt)
    outcome = OutcomeValue(
        spec=MODEL.configuration().target,
        status="observed",
        value=value,
        origin="synthetic",
        transform=None,
    )
    return ReplayFeedback(
        schema_version=1,
        lineage=record.lineage,
        target=record.opportunity.target,
        batch=FeedbackBatch(
            schema_version=1,
            decision_fingerprint=record.executed.fingerprint,
            feedback_id="constructed-supervised-label",
            revision_id=revision[0],
            supersedes=revision[1],
            horizon=MODEL.HORIZON,
            feedback=AggregateFeedback(values=(outcome,)),
            delay_rounds=delay,
            timing=FeedbackTiming(event_at=mature, available_at=mature, mature_at=mature, received_at=received),
        ),
        origins=(
            ValueOriginEvidence(
                member=None,
                dimension=outcome.spec.dimension,
                value_fingerprint=fingerprint(outcome.model_dump(mode="json")),
                origin="synthetic",
                producer=implementation(),
            ),
        ),
    )


class ForecastOwner:
    """Own one complete forecast replay for the existing state-store adapter.

    Parameters
    ----------
    child : int, optional
        Independent operational ownership coordinate; default0.
    """

    def __init__(self, child: int = 0):
        """Create a fresh host and the existing retained causal replay."""
        self.host = MODEL.ForecastHost(child)
        self.replay = RetainedReplay(
            self.host,
            ReplayPolicy(
                schema_version=1,
                stream=self.host.task,
                lineage=MODEL.reference(f"constructed-forecast-live-{child:02}"),
                parent=None,
                source=MODEL.SOURCE,
                capability=self.host.capability,
                delivery="delayed",
                specialists=False,
            ),
            state_type=MODEL.ForecastHostState,
            state_codec=MODEL.CODEC,
            retention=OrdinalRetention(profile="quiescent-ordinal-v1", source_epoch=EPOCH, maximum_tail_records=128),
        )
        self.replay.admit_history(history())

    def export_state(self):
        """Export the complete actual predictor and pending-label lineage.

        Returns
        -------
        RetainedContinuation
            Real schema2 retained replay and concrete host state.
        """
        return self.replay.export_continuation()

    def restore_state(self, state, /) -> None:
        """Restore only the explicitly chosen compatible host interpretation.

        Parameters
        ----------
        state : RetainedContinuation
            Complete original source, causal prefix, predictor and pending state.
        """
        self.replay = RetainedReplay.from_continuation(
            self.host, state, state_type=MODEL.ForecastHostState, state_codec=MODEL.CODEC
        )


def emit(owner: ForecastOwner, index: int):
    """Run an actual closed-drain prediction at the source-owned opportunity.

    Parameters
    ----------
    owner : ForecastOwner
        Actual independent consumer and causal replay.
    index : int
        Source-owned opportunity ordinal.

    Returns
    -------
    ReplayDecisionRecord
        Original executed prediction; no external action occurs.
    """
    return owner.replay.emit(
        owner.replay.prepare(opportunity(owner.host, owner.replay, index)), execution="offline_identity_v1"
    )


def run_prediction() -> dict[str, object]:
    """Execute chronological prediction against matched fixed/mean/last controls.

    Returns
    -------
    dict
        Human-readable raw per-step errors and complete state fingerprints.
        All controls use the same eligible prefix and zero tuning trials.
    """
    owner = ForecastOwner()
    rows = []
    for index, value in enumerate(LABELS):
        record = emit(owner, index)
        pending = owner.host.export_state().pending[-1]
        rows.append(
            {
                "step": index,
                "eligible_samples": len(owner.host.export_state().observations),
                "prediction": pending.prediction.value.value,
                "fixed": 0.0,
                "mean": pending.mean_control,
                "last": pending.last_control,
                "target": value,
            }
        )
        owner.replay.admit_feedback((feedback(record, value),))
    owner.replay.advance(eligible_through=instant(8), apply_at=instant(8))
    errors = {
        key: sum((row[key] - row["target"]) ** 2 for row in rows) / len(rows)
        for key in ("prediction", "fixed", "mean", "last")
    }
    return {
        "scope": "constructed-chronological-mechanics",
        "rows": rows,
        "squared_error_unit": "1",
        "mean_squared_errors": errors,
        "tuning_trials_each": 0,
        "lambda_development_declared": 1.0,
        "samples": len(owner.host.export_state().observations),
        "state_fingerprint": owner.export_state().fingerprint,
        "predictive_uncertainty": "not_qualified",
        "inference": "insufficient_evidence",
    }


def collection_layout(children: tuple[int, ...] = tuple(range(32))):
    """Declare real predictor/replay children using the existing owned store.

    Parameters
    ----------
    children : tuple of int, optional
        Unique ownership coordinates; default all32. This is operational
        recovery, without task pooling or independent-history claims.

    Returns
    -------
    tuple
        Existing typed collection definition and explicit concrete bindings.
    """
    if not children or len(set(children)) != len(children):
        raise ValueError("The explicit child inventory must be nonempty and unique")
    entries = []
    for child in sorted(children):
        host = MODEL.ForecastHost(child)
        outer = StateBinding(
            owner=host.task,
            implementation=MODEL.implementation(),
            state_codec=MODEL.reference("constructed-forecast-retained-continuation"),
            configuration_fingerprint=host.config.fingerprint,
        )
        codec = ComponentCodec(outer, RetainedContinuation[MODEL.ForecastHostState])
        entries.append(ComponentBinding(codec, partial(ForecastOwner, child), partial(child_frontier, child=host.task)))
    definition = CollectionDefinition(
        schema_version=1,
        reference=MODEL.reference("constructed-forecast-collection"),
        session=MODEL.reference("constructed-independent-forecast-sessions"),
        source_epoch=EPOCH,
        children=tuple(item.codec.binding.owner for item in entries),
        components=tuple(
            ComponentSpec(
                binding=item.codec.binding,
                child=item.codec.binding.owner,
                roles=("algorithm", "learner_pending", "delivery"),
                frontier_profile="replay-ordinal-v2",
            )
            for item in entries
        ),
    )
    return definition, tuple(entries)


def compatibility() -> ExecutionCompatibility:
    """Capture selected installed runtime and both public host source bindings.

    Returns
    -------
    ExecutionCompatibility
        Actual base distribution identity; source hashes are not authentication.
    """
    from coleman.spec.selected_runtime import SelectedInstalledRuntime  # pylint: disable=import-outside-toplevel

    return ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(
            MODEL.implementation(),
            implementation(),
        ),
    )


if __name__ == "__main__":
    print(json.dumps(run_prediction(), sort_keys=True, indent=2))
