"""Audited paired comparisons with verified records and explicit result availability."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Self

from pydantic import model_validator

from coleman.capabilities import require_capability
from coleman.evaluation.protected.audit import AccessReceipt, EvaluationAudit
from coleman.evaluation.protected.bounded_inference import BoundedMeanProtocol
from coleman.evaluation.protected.contracts import ComparisonHypothesis, EvaluationContract, FrozenEvaluation
from coleman.evaluation.protected.inference import (
    _bounded_equivalence,
    _compute_results,
    _supports_protocol,
    _validate_observations,
    holm_adjust,
)
from coleman.evaluation.protected.observations import ComparisonResult, ObservationReference, PairedObservation
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint


class FamilyReport(EvaluationContract):
    """Versioned report tied to an immutable freeze and pre-load access receipt.

    Receipt history determines confirmatory eligibility, independently of point
    estimates. This report never activates a policy. The frozen protocol declares
    the metric units, practical threshold, independent-unit meaning and assumptions.
    Digest-checked raw records are embedded and statistics recomputed on validation.
    ``completed_at`` is UTC calculation/source-assembly completion, after access
    and input availability; derived records retain this availability timestamp.
    """

    frozen: FrozenEvaluation
    access: AccessReceipt
    results: tuple[ComparisonResult, ...]
    inputs: tuple[ObservationReference, ...]
    completed_at: Timestamp

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Require exact freeze/scope/data identity and the complete predeclared family."""
        if (
            self.access.evaluation_fingerprint != self.frozen.fingerprint
            or self.access.holdout_id != self.frozen.protocol.holdout_id
            or self.access.data_fingerprint != self.frozen.data_fingerprint
        ):
            raise ValueError("Report receipt does not match its frozen evaluation")
        if tuple(item.hypothesis_id for item in self.results) != tuple(
            item.hypothesis_id for item in self.frozen.comparison_family
        ):
            raise ValueError("Report must contain the complete frozen comparison family")
        references = self._input_references()
        for hypothesis, result in zip(self.frozen.comparison_family, self.results, strict=True):
            self._check_references(hypothesis, result, references)
        return self

    def _input_references(self) -> dict[tuple[str, str], ObservationReference]:
        """Index unique, temporally eligible frozen candidate/unit references."""
        references: dict[tuple[str, str], ObservationReference] = {}
        record_keys: set[tuple[str, str, str, str]] = set()
        for reference in self.inputs:
            self.frozen.candidate(reference.candidate_id)
            key = reference.candidate_id, reference.metadata.group_id
            if key in references:
                raise ValueError("Duplicate report input unit/candidate")
            decision = self.frozen.plan.decide(reference.metadata)
            if decision.partition != "test" or not decision.eligible:
                raise ValueError("Ineligible report input metadata")
            record_key = reference.record_key
            if record_key in record_keys:
                raise ValueError("Repeated raw result identity across independent units")
            record_keys.add(record_key)
            references[key] = reference
        return references

    @staticmethod
    def _check_references(
        hypothesis: ComparisonHypothesis,
        result: ComparisonResult,
        references: dict[tuple[str, str], ObservationReference],
    ) -> None:
        """Match result counts and paired metadata to immutable raw-input references."""
        for candidate_id in (hypothesis.candidate_id, hypothesis.baseline_id):
            units = {unit for (name, unit), _reference in references.items() if name == candidate_id}
            if units != set(result.independent_units):
                raise ValueError("Report unit counts do not match its raw-input references")
        for unit in result.independent_units:
            left = references[hypothesis.candidate_id, unit]
            right = references[hypothesis.baseline_id, unit]
            if left.metadata != right.metadata:
                raise ValueError("Report input metadata does not match across paired units")

    @model_validator(mode="after")
    def _inference(self) -> Self:
        """Reject forged confirmatory claims on reused scopes or unsupported assumptions."""
        adjusted = holm_adjust(
            tuple(result.p_value if result.p_value is not None else 1.0 for result in self.results),
            outward=isinstance(self.frozen.protocol, BoundedMeanProtocol),
        )
        for result, corrected in zip(self.results, adjusted, strict=True):
            expected = corrected if result.p_value is not None else None
            if result.adjusted_p_value != expected:
                raise ValueError("Adjusted p-value differs from the complete frozen Holm family")
            self._check_point_estimate(result)
            if result.conclusion in {"evidence_of_benefit", "evidence_of_harm", "evidence_of_practical_equivalence"}:
                self._check_claim(result)
        return self

    def _check_point_estimate(self, result: ComparisonResult) -> None:
        """Require signed point estimates for uncertainty/claims and interval consistency."""
        mean, improvement = result.raw_mean_difference, result.improvement
        if mean is None or improvement is None:
            uncertainty = (result.ci_lower, result.ci_upper, result.p_value, result.adjusted_p_value)
            if any(value is not None for value in uncertainty) or result.conclusion in {
                "evidence_of_benefit",
                "evidence_of_harm",
                "evidence_of_practical_equivalence",
            }:
                raise ValueError("Uncertainty and confirmatory claims require non-null point estimates")
            return
        expected = mean if self.frozen.protocol.direction == "maximize" else -mean
        if not math.isclose(improvement, expected, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("Improvement does not match the protocol-defined raw difference sign")
        if (
            result.ci_lower is not None
            and result.ci_upper is not None
            and not result.ci_lower <= improvement <= result.ci_upper
        ):
            raise ValueError("Point estimate is outside its confidence interval")

    def _check_claim(self, result: ComparisonResult) -> None:
        """Validate access, independent observations and family-wise practical evidence."""
        protocol = self.frozen.protocol
        if not self.confirmatory_eligible or not _supports_protocol(protocol):
            raise ValueError("Unsupported confirmatory access or assumptions")
        if result.observed_pairs != len(result.independent_units) or result.observed_pairs < protocol.minimum_units:
            raise ValueError("Insufficient independent observations for confirmatory conclusion")
        if result.adjusted_p_value is None or result.ci_lower is None or result.ci_upper is None:
            raise ValueError("Unsupported confirmatory uncertainty")
        self._check_practical_claim(result, result.ci_lower, result.ci_upper, result.adjusted_p_value)

    def _check_practical_claim(self, result: ComparisonResult, lower: float, upper: float, corrected: float) -> None:
        """Validate practical benefit/harm or equivalence without changing legacy t conclusions."""
        protocol = self.frozen.protocol
        if result.conclusion == "evidence_of_practical_equivalence":
            if not _bounded_equivalence(protocol, lower, upper):
                raise ValueError("Unsupported practical equivalence interval or inference method")
        elif corrected > protocol.alpha:
            raise ValueError("Unsupported confirmatory uncertainty")
        if result.conclusion == "evidence_of_benefit" and lower <= protocol.practical_effect:
            raise ValueError("Benefit does not exceed the practical threshold")
        if result.conclusion == "evidence_of_harm" and upper >= -protocol.practical_effect:
            raise ValueError("Harm does not exceed the practical threshold")

    @model_validator(mode="after")
    def _verified_statistics(self) -> Self:
        """Recompute every estimate from digest-checked embedded records, without I/O.

        Student-t comparisons retain relative 1e-12/absolute 1e-15 numerical-
        library tolerance. Bounded rational calculations require identical binary64
        outputs, including outward endpoints. Discrete conclusions, reasons and
        counts must match exactly; threshold crossings fail closed.
        Completion cannot predate the access or any consumed raw outcome.
        """
        if self.completed_at < self.access.requested_at or any(
            self.completed_at < reference.record.available_at for reference in self.inputs
        ):
            raise ValueError("Report completion predates access or input availability")
        rows = tuple(
            PairedObservation(candidate_id=reference.candidate_id, metadata=reference.metadata, record=reference.record)
            for reference in self.inputs
        )
        observations = _validate_observations(self.frozen, rows)
        expected = _compute_results(self.frozen, observations, self.access)
        for actual, recomputed in zip(self.results, expected, strict=True):
            _require_matching_statistics(
                actual, recomputed, exact=isinstance(self.frozen.protocol, BoundedMeanProtocol)
            )
        return self

    @property
    def confirmatory_eligible(self) -> bool:
        """Return whether this API recorded the first protected confirmatory access."""
        return (
            self.access.partition == "test"
            and self.access.purpose == "confirmatory"
            and self.access.prior_test_accesses == 0
        )

    def domain_records(self, run_id: str, *, namespace: str = "ci.protected_comparisons") -> tuple[DomainRecord, ...]:
        """Convert estimates to #251 records, retaining nulls, signs, units and provenance.

        Each access/hypothesis has a distinct logical decision identity; repeating
        serialization or sink publication is idempotent. Unknown uncertainty is
        explicitly unavailable. Access IDs/sequences never count as independent
        evidence. No transformed learning reward is created from inference.
        """
        validated = type(self).model_validate(self)
        unit = validated.frozen.protocol.unit
        output = []
        for result in validated.results:
            quantities = (
                ("raw_mean_difference", result.raw_mean_difference),
                ("improvement", result.improvement),
                ("ci_lower", result.ci_lower),
                ("ci_upper", result.ci_upper),
            )
            measurements = tuple(
                Measurement(
                    name=name,
                    value=value,
                    unit=unit,
                    meaning=f"{validated.frozen.protocol.method}:{name}",
                    direction="none",
                    status="observed" if value is not None else "unavailable",
                )
                for name, value in quantities
            )
            measurements += (
                Measurement(
                    name="adjusted_p_value",
                    value=result.adjusted_p_value,
                    unit="1",
                    meaning="holm-family-wise-p-value-v1",
                    direction="none",
                    status="observed" if result.adjusted_p_value is not None else "unavailable",
                ),
            )
            measurements += (
                Measurement(
                    name="independent_units",
                    value=float(len(result.independent_units)),
                    unit="count",
                    meaning=validated.frozen.protocol.independent_unit,
                    direction="none",
                    status="observed",
                    minimum=0.0,
                ),
            )
            output.append(
                DomainRecord(
                    schema_version=1,
                    namespace=namespace,
                    run_id=run_id,
                    execution_id=validated.frozen.fingerprint,
                    decision_id=fingerprint([validated.access.sequence, result.hypothesis_id]),
                    data_fingerprint=validated.frozen.data_fingerprint,
                    available_at=validated.completed_at,
                    provenance=ResultProvenance(
                        producer="coleman.protected-evaluation",
                        producer_version="1",
                        spec_fingerprint=validated.frozen.fingerprint,
                        evaluation_fingerprint=validated.frozen.fingerprint,
                    ),
                    measurements=measurements,
                )
            )
        return tuple(output)


def _require_matching_statistics(actual: ComparisonResult, expected: ComparisonResult, *, exact: bool = False) -> None:
    """Reject statistics inconsistent with raw records and the frozen inference method.

    Legacy Student-t reports retain their specified numerical-library tolerance.
    The rational bounded method requires identical binary64 endpoints/probabilities;
    tolerant reads could admit an incorrectly narrowed conservative interval.
    """
    discrete_fields = ("hypothesis_id", "independent_units", "observed_pairs", "conclusion", "reason")
    if any(getattr(actual, field) != getattr(expected, field) for field in discrete_fields):
        raise ValueError("Report statistics differ from recomputed raw-record evidence")
    numeric_fields = ("raw_mean_difference", "improvement", "ci_lower", "ci_upper", "p_value", "adjusted_p_value")
    for field in numeric_fields:
        first, second = getattr(actual, field), getattr(expected, field)
        if first is None or second is None:
            matches = first is None and second is None
        elif exact:
            matches = first.hex() == second.hex()
        else:
            matches = math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-15)
        if not matches:
            raise ValueError("Report statistics differ from recomputed raw-record evidence")


def evaluate(
    frozen: FrozenEvaluation,
    audit: EvaluationAudit,
    load_observations: Callable[[], tuple[PairedObservation, ...]],
    *,
    actor: str,
) -> FamilyReport:
    """Reserve protected access, load typed records and evaluate the full frozen family.

    Legacy paired t inference requires independent units and normal differences.
    The explicitly versioned bounded-mean method instead requires known metric
    bounds and independent units; normality is not assumed. Unsupported legacy
    assumptions raise ``NotImplementedError`` before outcome loading. The new
    protocol rejects serial/clustered declarations at construction. Complete pairing is required;
    repeated seed rows are rejected. Missing outcomes produce insufficient evidence
    rather than selective complete-case tests. Holm p correction and Bonferroni
    simultaneous intervals use the entire predeclared family. Reuse is durably
    recorded and prevents a confirmatory conclusion, even after candidate changes.
    New-method practical equivalence uses an entire simultaneous interval within
    the predeclared region; it does not require rejecting a zero-difference null.
    Loader/validation failures leave access reservations visible for honest retries.
    """
    frozen = FrozenEvaluation.model_validate(frozen)
    if not _supports_protocol(frozen.protocol):
        raise NotImplementedError("Paired t inference requires independent units and declared normal differences")
    if not isinstance(frozen.protocol, BoundedMeanProtocol):
        require_capability("analytics")
    receipt = audit.access(frozen, partition="test", purpose="confirmatory", actor=actor)
    observations = _validate_observations(frozen, load_observations())
    updated = _compute_results(frozen, observations, receipt)
    references = tuple(
        ObservationReference(
            candidate_id=candidate_id,
            metadata=row.metadata,
            record_key=row.record.logical_key,
            record_digest=row.record.digest,
            record=row.record,
        )
        for candidate_id, units in sorted(observations.items())
        for _, row in sorted(units.items())
    )
    return FamilyReport(
        schema_version=1,
        frozen=frozen,
        access=receipt,
        results=updated,
        inputs=references,
        completed_at=datetime.now(UTC),
    )
