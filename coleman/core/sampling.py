"""Typed declarations of actual selection mass, separate from allocation weights."""

from fractions import Fraction
from typing import Literal, Self

from pydantic import model_validator

from coleman.core.actions import Action
from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, Identifier


class SamplingRecord(CoreImmutable):
    """Bind a mechanism's declared sampling evidence to exact action content.

    Attributes
    ----------
    mechanism : {'deterministic', 'sampled'}
        How the actual action was selected, independent from evidence status.
    mechanism_ref : Reference
        Explicit versioned selection implementation, not an inferred algorithm.
    status : str
        Exact mass, qualified approximation, unavailable or not required.
    action_fingerprint : str
        Content digest of the exact action this declaration describes.
    probability : float or None
        Required finite action mass in (0, 1] for exact/approximate evidence.
        Explicit null otherwise; zero observed-action mass is invalid.
    error_bound : float or None, optional
        Required absolute probability error in (0, 1] for approximation only.
    qualification : Reference or None, optional
        Required approximation qualification identity; not an executed proof.
    reason : str or None, optional
        Required stable code for unavailable evidence only.

    Notes
    -----
    This record does not sample, authenticate a declaration or derive a mass
    from ranks, scores, rewards or mixture weights. Continuous densities require
    a different contract. Strict status invariants reject ambiguous nulls.
    """

    mechanism: Literal["deterministic", "sampled"]
    mechanism_ref: Reference
    status: Literal["exact", "qualified_approximation", "unavailable", "not_required"]
    action_fingerprint: Fingerprint
    probability: Binary64 | None
    error_bound: Binary64 | None = None
    qualification: Reference | None = None
    reason: Identifier | None = None

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Check status-dependent nullable fields without inventing evidence."""
        if self.status in {"exact", "qualified_approximation"}:
            self._mass()
        elif self.probability is not None:
            raise ValueError("Unavailable/not-required sampling needs explicit null probability")
        if self.status == "qualified_approximation":
            if self.error_bound is None or not 0 < self.error_bound <= 1 or self.qualification is None:
                raise ValueError("Approximation requires an absolute error bound and qualification")
        elif self.error_bound is not None or self.qualification is not None:
            raise ValueError("Only qualified approximations may declare error/qualification")
        if (self.status == "unavailable") != (self.reason is not None):
            raise ValueError("Only unavailable sampling requires a reason")
        return self

    def _mass(self) -> None:
        """Require positive observed mass and a coherent deterministic declaration."""
        if self.probability is None or not 0 < self.probability <= 1:
            raise ValueError("Observed action mass must be finite and in (0, 1]")
        if self.mechanism == "deterministic" and self.status == "exact" and self.probability != 1:
            raise ValueError("Deterministic exact action mass must equal one")
        if (
            self.mechanism == "deterministic"
            and self.status == "qualified_approximation"
            and self.error_bound is not None
            and 1 - Fraction.from_float(self.probability) > Fraction.from_float(self.error_bound)
        ):
            raise ValueError("Deterministic approximate mass must bound its known probability of one")


def validate_sampling(action: Action, record: SamplingRecord) -> None:
    """Check action binding and finite-mass interpretation without sampling.

    Parameters
    ----------
    action : Action
        Actual validated action whose canonical content is referenced.
    record : SamplingRecord
        Mechanism declaration; exact and approximate mass support only finite
        choice, ranking, subset and slate actions.

    Raises
    ------
    ValueError
        Action digest differs or mass is attached to an unsupported action kind.

    Notes
    -----
    Passing these checks proves structural consistency, not sampler honesty,
    numerical support, calibration or an estimator's scientific assumptions.
    """
    checked = SamplingRecord.model_validate(record)
    if checked.action_fingerprint != action.fingerprint:
        raise ValueError("Sampling evidence is bound to different action content")
    if checked.status in {"exact", "qualified_approximation"} and action.kind not in {
        "choice",
        "ranking",
        "subset",
        "slate",
    }:
        raise ValueError("This contract does not interpret allocation/density/acquisition mass")
