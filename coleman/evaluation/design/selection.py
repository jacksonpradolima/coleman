"""Keep complete development selection and protected physical attempt ledgers distinct."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.primitives import Fingerprint, Identifier
from coleman.core.temporal import Timestamp
from coleman.evaluation.design._wire import DesignContract, wire_size
from coleman.evaluation.design.contracts import CandidatePlan, Seed
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.objectives.bindings import DefinitionBinding

if TYPE_CHECKING:
    from coleman.evaluation.design.contracts import ExperimentDesign


class CellIdentity(CoreImmutable):
    """Identify a logical result coordinate without calling it a physical job.

    Attributes
    ----------
    candidate_id, member_id : str
        Declared candidate and source member identities.
    seed : int
        Algorithmic repetition coordinate, never an inferential unit.
    """

    candidate_id: Identifier
    member_id: Identifier
    seed: Seed


class ConfigurationTrial(CoreImmutable):
    """Retain one attempted/inspected base configuration, including nonfinalists.

    Attributes
    ----------
    trial_id : str
        Distinct configuration-trial identity; retries belong to physical attempts.
    candidate : CandidatePlan
        Complete BASE configuration and frozen-law declarations actually inspected.
    status : str
        Declared/completed/failed/aborted/unavailable/inspected configuration state.
        A completed negative outcome is not a failed execution.
    partition : str
        Declared source partition used; protected/unknown exposure remains visible.
    completed_at : datetime or None
        Actual supplied trial/inspection completion, or explicit unknown.
    results : tuple of OriginalRecordEvidence
        Original development outcomes, if supplied. Omissions and nulls survive.
    source_accesses : tuple of str
        Actual known receipt fingerprints; missing history is not invented.
    reason : str or None, optional
        Supplied enclosing status explanation. Absence stays absent.
    """

    trial_id: Identifier
    candidate: CandidatePlan
    status: Literal["declared", "completed", "failed", "aborted", "unavailable", "inspected"]
    partition: Literal["train", "validation", "test", "unknown"]
    completed_at: Timestamp | None
    results: Annotated[tuple[OriginalRecordEvidence, ...], Field(max_length=8192)] = ()
    source_accesses: Annotated[tuple[Fingerprint, ...], Field(max_length=512)] = ()
    reason: Identifier | None = None

    @model_validator(mode="after")
    def _producer(self) -> Self:
        """Bind supplied development outcomes to the actual inspected base config."""
        expected = self.candidate.candidate.configuration_fingerprint
        if any(item.record.provenance.spec_fingerprint != expected for item in self.results):
            raise ValueError("Development outcome differs from its actual base candidate configuration")
        keys = [item.record_key for item in self.results]
        if len(keys) != len(set(keys)):
            raise ValueError("Development outcome evidence contains a duplicate original record")
        return self


class PhysicalAttempt(CoreImmutable):
    """Retain one actual grouped execution/retry and its own measured costs.

    Attributes
    ----------
    attempt_id, group_id : str
        Physical invocation and stable work-group identities. A group can emit
        many cells; a resumed logical cell can have several physical attempts.
    trial_id : str or None
        Known development-trial link; protected production may explicitly omit it.
    candidate_id : str
        Exact producing base candidate, checked by the enclosing design/panel.
    state : str
        Completed/failed/aborted/unavailable execution state, not scientific sign.
    started_at, completed_at : datetime or None
        Actual supplied physical times; unknown values stay explicit nulls.
    outputs : tuple of CellIdentity
        Exact logical output references; no one-process-per-cell assumption.
    costs : tuple of OriginalRecordEvidence
        Separate actual cost evidence. Empty means no supplied measurements,
        never free execution. Costs are not averaged or paid twice by the panel.
    """

    attempt_id: Identifier
    group_id: Identifier
    trial_id: Identifier | None
    candidate_id: Identifier
    state: Literal["completed", "failed", "aborted", "unavailable"]
    started_at: Timestamp | None
    completed_at: Timestamp | None
    outputs: Annotated[tuple[CellIdentity, ...], Field(max_length=16384)] = ()
    costs: Annotated[tuple[OriginalRecordEvidence, ...], Field(max_length=64)] = ()

    @model_validator(mode="after")
    def _attempt(self) -> Self:
        """Keep actual chronology/output ownership and cost identities consistent."""
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("Physical attempt completion precedes its actual start")
        keys = [(item.candidate_id, item.member_id, item.seed) for item in self.outputs]
        if len(keys) != len(set(keys)) or any(item.candidate_id != self.candidate_id for item in self.outputs):
            raise ValueError("Physical attempt outputs require unique coordinates of its actual candidate")
        if len({item.record_key for item in self.costs}) != len(self.costs):
            raise ValueError("A physical cost record cannot be charged twice in one attempt")
        return self


class AttemptLedger(DesignContract):
    """Retain actual development or protected work without mixing their roles.

    Attributes
    ----------
    stage : {'development', 'protected'}
        Selection contains development only. RawPanel owns its separate actual
        protected ledger; protected execution is not a new configuration trial.
    trials : tuple of ConfigurationTrial
        All supplied development configurations, including failed/nonfinalist ones.
        Protected ledgers require an empty trial catalogue.
    attempts : tuple of PhysicalAttempt
        Actual grouped invocations/retries, with costs even when no output exists.
        Every development invocation must retain its actual trial and the same
        producing candidate ID; a trial link cannot relabel another method's work.

    Notes
    -----
    Internal uniqueness does not discover undeclared external attempts. Importing
    outcomes does not fabricate physical work. The enclosing profile checks actual
    group/output counts and metadata bytes before publication.
    """

    stage: Literal["development", "protected"]
    trials: Annotated[tuple[ConfigurationTrial, ...], Field(max_length=256)] = ()
    attempts: Annotated[tuple[PhysicalAttempt, ...], Field(max_length=4096)] = ()

    @model_validator(mode="after")
    def _ledger(self) -> Self:
        """Reject hidden duplicate attempts/trials and double-charged physical cost."""
        trials = {item.trial_id for item in self.trials}
        if len(trials) != len(self.trials) or len({item.attempt_id for item in self.attempts}) != len(self.attempts):
            raise ValueError("Attempt ledger contains duplicate trial or physical-attempt identities")
        if self.stage == "protected" and self.trials:
            raise ValueError("Protected actual work cannot add development selection trials")
        if self.stage == "development" and any(item.trial_id not in trials for item in self.attempts):
            raise ValueError("Development physical attempts require their retained actual trial")
        if self.stage == "development":
            candidates = {item.trial_id: item.candidate.candidate.candidate_id for item in self.trials}
            if any(item.trial_id is None or item.candidate_id != candidates[item.trial_id] for item in self.attempts):
                raise ValueError("Development physical attempt candidate differs from its retained trial")
        costs = [cost.record_key for item in self.attempts for cost in item.costs]
        if len(costs) != len(set(costs)):
            raise ValueError("A physical cost record cannot be charged to several attempts")
        object.__setattr__(self, "trials", tuple(sorted(self.trials, key=lambda item: item.trial_id)))
        object.__setattr__(self, "attempts", tuple(sorted(self.attempts, key=lambda item: item.attempt_id)))
        return self

    @property
    def output_references(self) -> int:
        """Count actual output references, preserving retries without extra units."""
        return sum(len(item.outputs) for item in self.attempts)


class SelectionManifest(DesignContract):
    """Bind completed development selection to a design without a digest cycle.

    Attributes
    ----------
    design_fingerprint : str
        Exact complete design containing BASE configurations.
    ledger : AttemptLedger
        Development-only attempts/trials; protected work belongs to RawPanel.
    selected_candidate_ids : tuple of str
        Complete final design family, including baselines. No post-outcome family
        reduction is permitted; nonfinalist development trials stay in ledger.
    rule : DefinitionBinding
        Frozen validation selection rule, not a claim of authenticated timing.
    actual_selected_at : datetime
        Supplied actual completion instant, distinct from historical source cutoff.

    Notes
    -----
    The adapter persists this immutable manifest before protected disclosure and
    verifies actual receipt order. Caller timestamps alone cannot establish that
    outcomes were unseen. No trial, source access or selection runs here.
    """

    design_fingerprint: Fingerprint
    ledger: AttemptLedger
    selected_candidate_ids: Annotated[tuple[Identifier, ...], Field(min_length=2, max_length=64)]
    rule: DefinitionBinding
    actual_selected_at: Timestamp

    @field_validator("selected_candidate_ids")
    @classmethod
    def _selected(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize the complete final family without removing duplicates."""
        if len(values) != len(set(values)):
            raise ValueError("Selected final candidate IDs must be distinct")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _selection(self) -> Self:
        """Keep actual selection after every supplied development observation."""
        if self.ledger.stage != "development":
            raise ValueError("Selection requires development evidence, not protected actual work")
        times = [item.completed_at for item in self.ledger.trials if item.completed_at is not None]
        times.extend(record.record.available_at for item in self.ledger.trials for record in item.results)
        times.extend(item.completed_at for item in self.ledger.attempts if item.completed_at is not None)
        times.extend(item.started_at for item in self.ledger.attempts if item.started_at is not None)
        times.extend(cost.record.available_at for item in self.ledger.attempts for cost in item.costs)
        if any(moment > self.actual_selected_at for moment in times):
            raise ValueError("Selection precedes supplied actual development completion or outcome availability")
        return self

    @property
    def potential_test_exposure(self) -> bool:
        """Retain declared protected/unknown development inspection conservatively."""
        return any(item.partition in {"test", "unknown"} for item in self.ledger.trials)

    def require_design(self, design: ExperimentDesign) -> None:
        """Check exact design/family and the complete retained development budget.

        Parameters
        ----------
        design : ExperimentDesign
            Actual supplied design, never a lookup inferred from a human ID.

        Raises
        ------
        ValueError
            Changed design, dropped final family, exceeded attempt/metadata budget
            or incompatible complete base configuration. No I/O occurs.
        """
        expected = tuple(item.candidate.candidate_id for item in design.candidates)
        if self.design_fingerprint != design.fingerprint or self.selected_candidate_ids != expected:
            raise ValueError("Selection must bind the exact design and complete final candidate family")
        if len(self.ledger.trials) > design.budget.maximum_trials:
            raise ValueError("Complete development trials exceed the declared budget")
        if len(self.ledger.attempts) > design.budget.maximum_physical_attempts:
            raise ValueError("Actual development attempts exceed the declared budget")
        if self.ledger.output_references > design.budget.maximum_output_references:
            raise ValueError("Actual development output references exceed the declared budget")
        if any(
            len(trial.candidate.candidate.configuration_json.encode("utf-8")) > design.limits.max_configuration_bytes
            for trial in self.ledger.trials
        ):
            raise ValueError("Development trial configuration exceeds its declared capacity")
        if wire_size(self) + 1 > design.limits.max_selection_bytes:
            raise ValueError("Actual complete selection/ledger exceeds its declared metadata capacity")
