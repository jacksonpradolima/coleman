"""Content-bound definitions and explicit quantity interpretations without I/O."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.feedback import OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp

Interpretation = Literal[
    "raw_outcome",
    "resource",
    "priority_score",
    "probability_estimate",
    "monetary_amount",
    "risk_measure",
    "normalized_loss",
]


class DefinitionBinding(CoreImmutable):
    """Bind a semantic reference to the exact declared definition content.

    Attributes
    ----------
    reference : Reference
        Logical definition identity and semantic version.
    definition_fingerprint : str
        SHA-256 of the full versioned definition, including its parameters and
        supplied lineage. A digest identifies content; it authenticates nothing.

    Notes
    -----
    No definition lookup, registry, migration or filesystem access occurs.
    Different parameters can retain an operator version but never this binding.
    """

    reference: Reference
    definition_fingerprint: Fingerprint


class DefinitionLineage(CoreImmutable):
    """Retain supplied source facts without inventing missing provenance.

    Attributes
    ----------
    source : Reference or None
        Declared source identity, or explicit null when unknown.
    source_fingerprint : str or None
        Declared source-content digest, independently unknown when null.
    declared_at : datetime or None
        Supplied zoned declaration instant, normalized to UTC; null is untimed.
    supersedes : DefinitionBinding or None
        Previous definition content, or explicit null for a root/unknown lineage.

    Notes
    -----
    These are attestations, not verified access, fit eligibility or authenticity.
    Construction reads no clock and does not supersede any stored definition.
    """

    source: Reference | None
    source_fingerprint: Fingerprint | None
    declared_at: Timestamp | None
    supersedes: DefinitionBinding | None


class SemanticDefinition(CoreContract):
    """Supply an immutable v1 definition independently of transport or runtime.

    Attributes
    ----------
    reference : Reference
        Caller-owned semantic identity and exact version.
    lineage : DefinitionLineage
        Explicit source/declaration facts, including unknowns.

    Notes
    -----
    Concrete subclasses declare all parameters as typed immutable fields. The
    inherited guarded codecs reject missing/future schema versions and unknown
    fields. No implicit migration, registration or execution occurs.
    """

    reference: Reference
    lineage: DefinitionLineage

    @property
    def binding(self) -> DefinitionBinding:
        """Return the reference and digest of this complete validated definition.

        Returns
        -------
        DefinitionBinding
            Content identity covering units, parameters and lineage.

        Raises
        ------
        ValueError, TypeError
            Unchecked nested content fails revalidation or canonical encoding.

        Notes
        -----
        Performs no lookup or I/O and does not authenticate source facts.
        """
        return DefinitionBinding(reference=self.reference, definition_fingerprint=self.fingerprint)


class QuantityMeaning(CoreImmutable):
    """Keep physical units, numerical role and scientific interpretation distinct.

    Attributes
    ----------
    spec : ScalarSpec
        Exact dimension/version, unit, meaning, role and inclusive value bounds.
    interpretation : str
        Raw outcome, resource, priority score, probability estimate, monetary
        amount, named risk measure or normalized learner loss.
    horizon : Reference or None
        Exact measurement horizon, or explicit null for a genuinely untimed use.
    risk_definition : Reference or None, optional
        Required exact definition for risk interpretation; forbidden otherwise.

    Raises
    ------
    ValueError
        Incompatible interpretation/role, malformed probability/loss bounds or
        a missing/extraneous risk definition.

    Notes
    -----
    Probability estimates are not qualified as calibrated by a label. Loss and
    probability both use unit ``1`` but cannot substitute for each other. Opaque
    custom units remain valid raw declarations; conversion is a separate call.
    No transformation, inference, currency conversion or I/O occurs.
    """

    spec: ScalarSpec
    interpretation: Interpretation
    horizon: Reference | None
    risk_definition: Reference | None = None

    @model_validator(mode="after")
    def _interpretation(self) -> Self:
        """Reject role substitution and incomplete risk/normalized semantics."""
        roles = {
            "raw_outcome": {"objective", "diagnostic"},
            "resource": {"resource"},
            "priority_score": {"feature", "prediction", "diagnostic"},
            "probability_estimate": {"feature", "prediction", "diagnostic"},
            "monetary_amount": {"objective", "resource", "diagnostic"},
            "risk_measure": {"objective", "resource", "diagnostic"},
            "normalized_loss": {"loss"},
        }
        if self.spec.role not in roles[self.interpretation]:
            raise ValueError("Quantity interpretation is incompatible with its scalar role")
        if (self.interpretation == "risk_measure") != (self.risk_definition is not None):
            raise ValueError("Exactly risk quantities require a named risk definition")
        if self.interpretation in {"probability_estimate", "normalized_loss"}:
            self._bounded_dimensionless()
        return self

    def _bounded_dimensionless(self) -> None:
        """Require declared dimensionless bounds inside the closed unit interval."""
        if (
            self.spec.semantics.unit != "1"
            or self.spec.minimum is None
            or self.spec.maximum is None
            or not 0 <= self.spec.minimum <= self.spec.maximum <= 1
        ):
            raise ValueError("Probability/loss quantities require explicit dimensionless [0,1] bounds")


def require_quantity(value: Scalar, meaning: QuantityMeaning) -> Scalar:
    """Revalidate a scalar against its exact supplied quantity declaration.

    Parameters
    ----------
    value : Scalar
        Observed or explicitly missing quantity; caller data is never changed.
    meaning : QuantityMeaning
        Exact expected schema and interpretation.

    Returns
    -------
    Scalar
        Revalidated input with its original value/status and bounds.

    Raises
    ------
    ValueError
        Invalid content or mismatched dimension/version/role/units/meaning/bounds.

    Notes
    -----
    Performs no unit conversion, interpretation inference, imputation or I/O.
    The scalar alone does not carry horizon; outer operations bind that context.
    """
    checked = Scalar.model_validate(value)
    expected = QuantityMeaning.model_validate(meaning)
    if checked.spec.model_dump_json() != expected.spec.model_dump_json():
        raise ValueError("Quantity does not match the exact declared schema")
    return checked


def _project_loss_outcome(
    raw: OutcomeValue, expected_raw: Scalar, output: Scalar, transform: Reference
) -> OutcomeValue:
    """Bind checked operation output to matching raw content and retained origin."""
    original = OutcomeValue.model_validate(raw)
    scalar = Scalar(spec=original.spec, value=original.value, status=original.status)
    if scalar.model_dump_json() != expected_raw.model_dump_json():
        raise ValueError("Outcome does not match the exact operation raw content")
    return OutcomeValue(
        spec=output.spec,
        value=output.value,
        status=output.status,
        origin=original.origin,
        transform=transform,
    )
