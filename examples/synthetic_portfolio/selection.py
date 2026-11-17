"""Development-only tuning and protected descriptive access on one authored tape."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreImmutable
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint
from coleman.evaluation.protected.audit import AccessReceipt, EvaluationAudit
from coleman.evaluation.protected.contracts import (
    ComparisonHypothesis,
    EvaluationProtocol,
    FrozenCandidate,
    FrozenEvaluation,
    freeze,
)
from coleman.evaluation.protected.temporal import TemporalPlan
from coleman.problems.portfolio import Exact
from coleman.problems.portfolio.contracts import PortfolioContract

from . import model
from .experiment import Experiment, ExperimentState


class FrozenPortfolioConfiguration(CoreImmutable):
    """Type the fixed candidate configuration within the schema1 example.

    Attributes
    ----------
    strength : float
        Declared L1 strength in coefficient units, from zero through 100.
    warm_state : str
        Shared complete development-state content digest, not authentication.
    mode : str
        One original method from the fixed example catalogue.
    cash_reserve : str
        Required target cash fraction ``1/50`` of settled post-fee wealth.
    recipe_target_adapter, funded_profile : str
        Required versioned exact-coefficient and funded-long-only semantics.

    Notes
    -----
    The containing frozen-evaluation/result contracts supply the schema version.
    These fixed versioned adapter names and exact cash fraction describe the
    constructed long-only example, not arbitrary portfolio consumers. A warm
    digest identifies declared content without proving its origin or execution.
    """

    strength: Annotated[Binary64, Field(ge=0, le=100)]
    warm_state: Fingerprint
    mode: str
    cash_reserve: Literal["1/50"]
    recipe_target_adapter: Literal["exact-ratio-coefficients-v1"]
    funded_profile: Literal["funded-long-only-exact-v1"]

    @model_validator(mode="after")
    def _mode(self):
        """Reject an unsupported fixed-example method instead of relabelling it."""
        if self.mode not in model.MODES:
            raise ValueError("Frozen portfolio configuration requires a known original mode")
        return self


class DevelopmentTrial(PortfolioContract):
    """Keep each declared strength's own full funded development output."""

    strength: Annotated[Binary64, Field(gt=0, le=100)]
    state: ExperimentState
    net_value: Exact

    @model_validator(mode="after")
    def _original(self):
        """Verify raw score and exact state belong to the original declared trial."""
        if self.state.next_index != 6 or self.state.strength != self.strength:
            raise ValueError("Development trial must retain its original six-period selected law")
        actual = next(row for row in self.state.periods[-1].candidates if row.name == "anchored-bounded-interval-l1")
        if self.net_value != actual.settlement.net_value:
            raise ValueError("Development score differs from its original own-book output")
        return self


class DevelopmentSelection(PortfolioContract):
    """Record a fixed finite grid and development-only original stopping rule."""

    trials: Annotated[tuple[DevelopmentTrial, ...], Field(min_length=1, max_length=3)]
    selected_strength: Annotated[Binary64, Field(gt=0, le=100)]
    rule: Literal["greatest-six-period-own-book-net-then-smallest-strength-v1"] = (
        "greatest-six-period-own-book-net-then-smallest-strength-v1"
    )

    @model_validator(mode="after")
    def _selection(self):
        """Keep the predeclared grid and exact raw-score tie rule immutable."""
        if tuple(trial.strength for trial in self.trials) != (0.025, 0.05, 0.1):
            raise ValueError("Development selection requires the full predeclared positive-strength grid")
        best = min(self.trials, key=lambda trial: (-trial.net_value.fraction, trial.strength))
        if self.selected_strength != best.strength:
            raise ValueError("Development selection differs from its original declared tie rule")
        return self


class ProtectedResult(PortfolioContract):
    """Retain protected access with honest descriptive scope and fixed fallback."""

    selection_fingerprint: Fingerprint
    frozen: FrozenEvaluation
    access: AccessReceipt
    state: ExperimentState
    status: Literal["descriptive_only"] = "descriptive_only"
    reason: Literal["one-authored-dependent-tape-no-independent-inference"] = (
        "one-authored-dependent-tape-no-independent-inference"
    )
    fallback: Literal["equal"] = "equal"

    @model_validator(mode="after")
    def _bindings(self):
        """Bind the descriptive result to its exact frozen source and access."""
        if self.state.next_index != 16 or self.state.tape_fingerprint != self.frozen.data_fingerprint:
            raise ValueError("Protected result requires its complete original frozen source")
        if (
            self.access.evaluation_fingerprint != self.frozen.fingerprint
            or self.access.data_fingerprint != self.frozen.data_fingerprint
            or self.access.holdout_id != self.frozen.protocol.holdout_id
            or self.access.partition != "test"
            or self.access.purpose != "verification"
        ):
            raise ValueError("Protected result access differs from its actual frozen evaluation")
        if tuple(candidate.candidate_id for candidate in self.frozen.candidates) != tuple(sorted(model.MODES)):
            raise ValueError("Protected result lost a declared original comparison candidate")
        configurations = tuple(
            FrozenPortfolioConfiguration.model_validate_json(candidate.configuration_json)
            for candidate in self.frozen.candidates
        )
        if (
            any(
                configuration.mode != candidate.candidate_id or configuration.strength != self.state.strength
                for candidate, configuration in zip(self.frozen.candidates, configurations, strict=True)
            )
            or len({configuration.warm_state for configuration in configurations}) != 1
        ):
            raise ValueError("Protected result differs from its actual frozen portfolio configuration")
        return self


def develop() -> DevelopmentSelection:
    """Choose a positive L1 strength exclusively on the six-period prefix.

    Returns
    -------
    DevelopmentSelection
        All three actual own-book states and raw signed net values, including
        losing/tied trials. Selection maximizes full-combination funded value,
        then selects the smallest strength on an exact tie. No suffix row is
        executed or supplied to any learner, transform or predictor.

    Notes
    -----
    This tiny authored grid demonstrates protocol mechanics. It supplies no
    economic efficacy or independent observations and performs no I/O.
    """
    trials = []
    for strength in (0.025, 0.05, 0.1):
        experiment = Experiment(strength=strength)
        for index in range(6):
            experiment.step(index)
        state = experiment.export_state()
        result = next(row for row in state.periods[-1].candidates if row.name == "anchored-bounded-interval-l1")
        assert result.settlement.net_value is not None
        trials.append(
            DevelopmentTrial(schema_version=1, strength=strength, state=state, net_value=result.settlement.net_value)
        )
    selected = min(trials, key=lambda trial: (-trial.net_value.fraction, trial.strength))
    return DevelopmentSelection(schema_version=1, trials=tuple(trials), selected_strength=selected.strength)


def run_protected(selection: DevelopmentSelection, directory: Path) -> ProtectedResult:
    """Persist the chosen law and reserve actual access before suffix execution.

    Parameters
    ----------
    selection : DevelopmentSelection
        Complete actual six-period trials and fixed selected positive strength.
    directory : pathlib.Path
        Trusted artifact/audit root. Retrying reserves another visible protected
        access; no caller key disguises repeated inspection.

    Returns
    -------
    ProtectedResult
        Full comparison family, original audit access and16-period own books.
        Fixed equal-weight fallback remains selected for adoption. One authored
        serial tape cannot establish independent uncertainty or adaptive benefit.

    Raises
    ------
    ValueError, OSError
        Altered selection/frontier, incompatible state, refused component,
        conflicting artifact or failed durable access reservation. No live trades.

    Notes
    -----
    Uses the existing frozen-evaluation and audit contracts. The serial,
    nonnormal protocol is deliberately unsupported by the paired inference
    engine, which is therefore not called. Raw differences remain valid output.
    """
    chosen = DevelopmentSelection.model_validate(selection)
    trial = next((trial for trial in chosen.trials if trial.strength == chosen.selected_strength), None)
    if trial is None or any(item.state.next_index != 6 for item in chosen.trials):
        raise ValueError("Protected run requires its original six-period development selection")
    expected = min(chosen.trials, key=lambda item: (-item.net_value.fraction, item.strength))
    if expected.strength != chosen.selected_strength:
        raise ValueError("Protected run cannot replace the original development rule")
    experiment = Experiment(strength=trial.strength)
    experiment.restore_state(trial.state)
    warm_fingerprint = trial.state.fingerprint
    candidates = tuple(
        FrozenCandidate.from_configuration(
            name,
            FrozenPortfolioConfiguration(
                strength=trial.strength,
                warm_state=warm_fingerprint,
                mode=name,
                cash_reserve="1/50",
                recipe_target_adapter="exact-ratio-coefficients-v1",
                funded_profile="funded-long-only-exact-v1",
            ).model_dump(mode="json"),
        )
        for name in model.MODES
    )
    family = tuple(
        ComparisonHypothesis(hypothesis_id=name + "-vs-equal", candidate_id=name, baseline_id="equal")
        for name in model.MODES
        if name != "equal"
    ) + (
        ComparisonHypothesis(hypothesis_id="penalty-vs-cadence", candidate_id="penalized", baseline_id="cadence"),
        ComparisonHypothesis(hypothesis_id="penalty-vs-movement", candidate_id="penalized", baseline_id="movement"),
    )
    plan = TemporalPlan(
        train_end=model.START + timedelta(days=2),
        validation_end=model.START + timedelta(days=5),
        test_end=model.START + timedelta(days=15),
    )
    protocol = EvaluationProtocol(
        protocol_id="authored-portfolio-descriptive-v1",
        holdout_id="authored-portfolio-suffix-v1",
        metric_name="terminal_net_value",
        unit="synthetic-credit",
        meaning="own-funded-book-net-value-v1",
        direction="maximize",
        independent_unit="authored-path",
        dependence="serial",
        normal_differences_assumed=False,
        practical_effect=0.0,
    )
    frozen = freeze(candidates, trial.state.tape_fingerprint, plan, protocol, family)
    writer = ArtifactWriter(directory)
    writer.write_immutable(directory / "development-selection.json", chosen.to_json())
    writer.write_immutable(directory / "protected-freeze.json", frozen.to_json())
    access = EvaluationAudit(directory / "access-audit").access(
        frozen, partition="test", purpose="verification", actor="public-synthetic-example"
    )
    for index in range(6, 16):
        experiment.step(index)
    return ProtectedResult(
        schema_version=1,
        selection_fingerprint=chosen.fingerprint,
        frozen=frozen,
        access=access,
        state=experiment.export_state(),
    )
