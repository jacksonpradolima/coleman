"""Versioned terminal-allocation inputs and original purchased observations."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from coleman.core.actions import AcquisitionAction, AcquisitionSpace
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar, ScalarSpec
from coleman.core.primitives import Fingerprint, Identifier, canonical_json
from coleman.core.quantities import Quantity, decimal_text, ticks

Count = Annotated[int, Field(strict=True, ge=0, le=1024)]


def _canonical_quantity(value: Quantity) -> Quantity:
    """Retain exact numeric value/units with canonical Decimal quantum for state codecs."""
    return Quantity(value=Decimal(decimal_text(value.value)), semantics=value.semantics)


def _spec_content(value: ScalarSpec) -> str:
    """Compare exact declared specification bytes, retaining binary64 signed zero."""
    return canonical_json(value.model_dump(mode="json"))


ResourceQuantity = Annotated[Quantity, AfterValidator(_canonical_quantity)]


class Fidelity(CoreImmutable):
    """Name one cumulative resource target, without asserting predictive validity.

    Attributes
    ----------
    reference : Reference
        Versioned target identity; tuple order in the plan is significant.
    resource : Quantity
        Positive cumulative model/physical resource in the budget's exact units.
        Under stateless billing the whole target is paid on every rerun.
    """

    reference: Reference
    resource: ResourceQuantity


class Candidate(CoreImmutable):
    """Declare a candidate and optional shared setup charge group.

    Attributes
    ----------
    reference : Reference
        Complete admitted candidate identity, not a rank.
    setup_group : Reference or None
        Setup is quoted once when this group's first purchase is accepted.
        This limited interaction is explicit; arbitrary joint cost models are
        unsupported rather than assumed additive.
    """

    reference: Reference
    setup_group: Reference | None = None


class SetupCharge(CoreImmutable):
    """Declare one nonnegative once-per-run shared setup amount in budget units."""

    group: Reference
    resource: ResourceQuantity


class CalibrationCase(CoreImmutable):
    """Retain one training-only early-to-terminal comparison, including producer.

    Attributes
    ----------
    case_id : str
        Training case identity, not an independent-observation claim.
    producer : Reference
        Original observation producer retained by the qualification.
    early : tuple of Scalar
        Observed scores at every nonterminal fidelity, in declared order.
    terminal : Scalar
        Observed terminal score with exactly the same specification.
    """

    case_id: Identifier
    producer: Reference
    early: Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=9)]
    terminal: Scalar


class FidelityQualification(CoreContract):
    """Freeze an empirical training-only fidelity gate, not a general error bound.

    Attributes
    ----------
    training_partition, evaluation_partition : Reference
        Distinct declared partitions. Evaluation outcomes must never enter cases.
    training_source, evaluation_source : str
        Distinct content fingerprints selected before protected evaluation.
    maximum_error : float
        Nonnegative admitted absolute early-to-terminal discrepancy in score
        units; every supplied training pair must satisfy it.
    cases : tuple of CalibrationCase
        Complete predeclared finite training comparison set, not a fitted test
        set. A passing gate provides no guarantee on a new candidate/path.

    Notes
    -----
    Partition/source declarations do not authenticate an external producer.
    Host protected-access controls remain necessary. No tuning or I/O occurs.
    """

    training_partition: Reference
    evaluation_partition: Reference
    training_source: Fingerprint
    evaluation_source: Fingerprint
    maximum_error: Binary64
    cases: Annotated[tuple[CalibrationCase, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _training(self) -> Self:
        """Check the declared training gate without ever opening evaluation data."""
        if self.training_partition.id == self.evaluation_partition.id or self.training_source == self.evaluation_source:
            raise ValueError("Fidelity qualification needs distinct training and evaluation partitions/sources")
        if self.maximum_error < 0:
            raise ValueError("Fidelity error tolerance must be nonnegative")
        if len({row.case_id for row in self.cases}) != len(self.cases):
            raise ValueError("Duplicate training case")
        for case in self.cases:
            if case.terminal.status != "observed":
                raise ValueError("Qualification requires observed terminal training outcomes")
            for early in case.early:
                if early.status != "observed" or _spec_content(early.spec) != _spec_content(case.terminal.spec):
                    raise ValueError("Qualification requires comparable observed training pairs")
                if early.value is None or case.terminal.value is None:
                    raise ValueError("Observed qualification pair requires values")
                discrepancy = abs(Fraction(early.value) - Fraction(case.terminal.value))
                if discrepancy > Fraction(self.maximum_error):
                    raise ValueError("Training fidelity gate failed; do not eliminate on this proxy")
        return self


class AllocationPlan(CoreContract):
    """Freeze a small complete terminal best-candidate identification experiment.

    Attributes
    ----------
    reference : Reference
        Logical experiment and version; method changes require a new plan.
    source_fingerprint : str
        Full admitted evaluation pool identity, never a hidden truncation.
    candidates : tuple of Candidate
        One through 64 complete members, canonicalized by ID.
    fidelities : tuple of Fidelity
        One through 10 strictly increasing positive cumulative resource targets;
        final entry is terminal. Equal allocation uses terminal only.
    method : {'equal_allocation_v1', 'successive_halving_v1'}
        Deterministic terminal identification profiles; ties favor stable IDs.
    billing : {'incremental', 'stateless_rerun'}
        Incremental target difference or full target per acquisition.
    budget : Quantity
        Exact nonnegative resource budget, at most 10^12 units. Quotes prevent starting an acquisition
        that would exceed it; actual unknown or excessive charges stop the run.
    outcome : ScalarSpec
        Explicit terminal objective units/meaning; role must be objective.
    direction : {'minimize', 'maximize'}
        Terminal preference, independent of cumulative reward.
    setups : tuple of SetupCharge
        Optional explicit shared interactions, each paid at most once.
    qualification : FidelityQualification or None
        Required for every early elimination route; never used by equal allocation.

    Notes
    -----
    Plans hold no evaluation outcomes. No Hyperband, knapsack-bandit, cumulative
    regret, optimal information-value or universal proxy-validity claim applies.
    The finite limits bound replay, not a silently selected source projection.
    New resource fields retain exact decimal numeric values with canonical
    quantum; legacy quantity readers and coverage artifact meanings are unchanged.
    """

    reference: Reference
    source_fingerprint: Fingerprint
    candidates: Annotated[tuple[Candidate, ...], Field(min_length=1, max_length=64)]
    fidelities: Annotated[tuple[Fidelity, ...], Field(min_length=1, max_length=10)]
    method: Literal["equal_allocation_v1", "successive_halving_v1"]
    billing: Literal["incremental", "stateless_rerun"]
    budget: ResourceQuantity
    outcome: ScalarSpec
    direction: Literal["minimize", "maximize"]
    setups: Annotated[tuple[SetupCharge, ...], Field(max_length=64)] = ()
    qualification: FidelityQualification | None = None

    @model_validator(mode="after")
    def _plan(self) -> Self:
        """Reject unmatched units, incomplete memberships and unqualified proxies."""
        ordered_references(tuple(row.reference for row in self.candidates))
        ordered_references(tuple(row.reference for row in self.fidelities))
        ordered_references(tuple(row.group for row in self.setups))
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates, key=lambda row: row.reference.id)))
        object.__setattr__(self, "setups", tuple(sorted(self.setups, key=lambda row: row.group.id)))
        if self.outcome.role != "objective":
            raise ValueError("Terminal identification requires an objective specification")
        if self.budget.value > Decimal("1e12"):
            raise ValueError("Terminal allocation budget exceeds 10^12 declared units")
        self._resources()
        self._groups()
        self._qualification()
        return self

    def _resources(self) -> None:
        """Check strictly increasing exact resource targets against budget semantics."""
        previous = 0
        for fidelity in self.fidelities:
            current = ticks(fidelity.resource.value)
            if fidelity.resource.semantics != self.budget.semantics or current <= previous:
                raise ValueError("Fidelity targets must increase positively in exact budget units")
            previous = current

    def _groups(self) -> None:
        """Check the explicit complete shared-setup interaction declaration."""
        groups = {(row.group.id, row.group.version) for row in self.setups}
        if any(
            row.setup_group is not None and (row.setup_group.id, row.setup_group.version) not in groups
            for row in self.candidates
        ):
            raise ValueError("Candidate references an undeclared setup group")
        if groups != {
            (row.setup_group.id, row.setup_group.version) for row in self.candidates if row.setup_group is not None
        }:
            raise ValueError("Unused setup group is not part of this complete plan")
        if any(row.resource.semantics != self.budget.semantics for row in self.setups):
            raise ValueError("Shared setup charges require exact budget units")

    def _qualification(self) -> None:
        """Bind every early elimination to training-only source/schedule evidence."""
        early_route = self.method == "successive_halving_v1" and len(self.fidelities) > 1
        if early_route:
            qualification = self.qualification
            if qualification is None or qualification.evaluation_source != self.source_fingerprint:
                raise ValueError("Early elimination requires training-only qualification for this source")
            for case in qualification.cases:
                if len(case.early) != len(self.fidelities) - 1 or _spec_content(case.terminal.spec) != _spec_content(
                    self.outcome
                ):
                    raise ValueError("Training qualification does not cover this fidelity schedule/objective")
        elif self.qualification is not None:
            raise ValueError("Terminal-only allocation does not consume a proxy qualification")

    @property
    def space(self) -> AcquisitionSpace:
        """Return actual candidate/fidelity membership, without acquiring anything.

        Returns
        -------
        AcquisitionSpace
            Exact content-bound pool and explicit versioned targets.
        """
        return AcquisitionSpace(
            feasible_set=Reference(id=self.reference.id, version=self.fingerprint),
            members=tuple(row.reference for row in self.candidates),
            fidelities=tuple(row.reference for row in self.fidelities),
        )


class Purchase(CoreContract):
    """Bind a deterministic logical acquisition to its explicit resource quote.

    Attributes
    ----------
    purchase_id : str
        Content-bound logical identity. Retrying execution may still have real
        side effects; external producers must implement their own reconciliation.
    ordinal : int
        Zero-based ledger position; complete runs accept at most 640 acquisitions.
    action : AcquisitionAction
        Actual chosen candidate and cumulative fidelity target, not a ranking.
    prior_fidelity : Reference or None
        Last purchased target for this candidate; null on the first acquisition.
    quoted_charge : Quantity
        Requested increment/full rerun plus any first shared setup, in budget units.
    """

    purchase_id: Fingerprint
    ordinal: Count
    action: AcquisitionAction
    prior_fidelity: Reference | None
    quoted_charge: ResourceQuantity


class AcquisitionReceipt(CoreContract):
    """Record only a purchased observation and its producer's actual resource use.

    Attributes
    ----------
    purchase : Purchase
        Exact prior logical request, retained without substituted features.
    producer : Reference
        Original execution/observation source, independent of logical identity.
    outcome : Scalar
        Observed terminal/proxy score or explicit missingness; negatives and zero
        are valid if the declared objective permits them.
    charge : Quantity or None
        Actual metered resource, at most 10^12 declared units per receipt;
        null means unknown, never free.
    delay_steps : int
        Nonnegative producer-reported receipt delay in application steps. The
        sequential reference waits for this receipt before making another choice.
    """

    purchase: Purchase
    producer: Reference
    outcome: Scalar
    charge: ResourceQuantity | None
    delay_steps: Count

    @model_validator(mode="after")
    def _charge_limit(self) -> Self:
        """Keep finite ledger accounting inside its declared 10^12-per-receipt envelope."""
        if self.charge is not None and self.charge.value > Decimal("1e12"):
            raise ValueError("Actual acquisition charge exceeds 10^12 declared units")
        return self
