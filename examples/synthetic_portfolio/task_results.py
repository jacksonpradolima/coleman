"""Compact v2 publications reference complete original v1 ACK-backed evidence."""

from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.primitives import Fingerprint, canonical_json
from coleman.objectives.bindings import DefinitionBinding
from coleman.problems.portfolio import Exact
from coleman.problems.portfolio.contracts import PortfolioContract
from coleman.state.storage_contracts import StateBlobRef
from coleman.tasks.contracts import OwnedCommitProof

from . import model
from .experiment import ExperimentState, ExperimentStateV2


class PortfolioTerminalOutcome(PortfolioContract):
    """Summarize one complete own-book path in synthetic credits.

    Attributes
    ----------
    name : str
        Original candidate identity in the declared comparison catalogue.
    currency : {'synthetic-credit'}
        Unit of every wealth, charge and traded-notional quantity here.
    terminal_net_value : Exact
        Last post-fee funded wealth; a value below initial capital is valid.
    proportional_paid, recurring_paid : Exact
        Separate nonnegative charges summed over the complete original path.
    executed_full_l1_notional : Exact
        Sum of absolute executed credit notionals, without a one-half factor.

    Notes
    -----
    Paid-charge and full absolute executed-notional totals cover every original
    period. They exclude coefficient movement and mathematical penalties. The
    complete receipts, signed returns, coefficients and objectives remain in
    the referenced state artifact; these totals imply no financial inference.
    """

    name: str
    currency: Literal["synthetic-credit"] = "synthetic-credit"
    terminal_net_value: Exact
    proportional_paid: Exact
    recurring_paid: Exact
    executed_full_l1_notional: Exact

    @model_validator(mode="after")
    def _funded(self) -> Self:
        """Refuse negative funded wealth or negative paid charges/notional."""
        if any(
            value.numerator < 0
            for value in (
                self.terminal_net_value,
                self.proportional_paid,
                self.recurring_paid,
                self.executed_full_l1_notional,
            )
        ):
            raise ValueError("Funded terminal wealth, paid charges and absolute executed notional are nonnegative")
        return self


def terminal_outcomes(state: ExperimentState | ExperimentStateV2) -> tuple[PortfolioTerminalOutcome, ...]:
    """Project every candidate from a fully validated complete original path.

    Parameters
    ----------
    state : ExperimentState or ExperimentStateV2
        Complete16-period typed state. Every settlement must be executed.

    Returns
    -------
    tuple of PortfolioTerminalOutcome
        Original candidate order, terminal wealth and exact summed credit
        charges/full traded notional. No source is acquired or model updated.

    Raises
    ------
    ValueError
        Partial trajectory or unavailable/refused settlement.
    """
    if type(state) is ExperimentState:  # pylint: disable=unidiomatic-typecheck
        current = ExperimentState.model_validate(state)
    elif type(state) is ExperimentStateV2:  # pylint: disable=unidiomatic-typecheck
        current = ExperimentStateV2.model_validate(state)
    else:
        raise TypeError("Terminal summaries require an exact selected experiment state")
    if current.next_index != 16 or any(
        row.settlement.status != "executed" for period in current.periods for row in period.candidates
    ):
        raise ValueError("Terminal summaries require every executed original period")
    result = []
    for index, name in enumerate(model.MODES):
        receipts = tuple(period.candidates[index].settlement for period in current.periods)
        last = receipts[-1].net_value
        assert last is not None
        totals = []
        for field in ("proportional_paid", "recurring_paid", "turnover"):
            values = tuple(getattr(receipt, field) for receipt in receipts)
            assert all(value is not None for value in values)
            totals.append(Exact.of(sum(value.fraction for value in values if value is not None)))
        result.append(
            PortfolioTerminalOutcome(
                schema_version=1,
                name=name,
                terminal_net_value=last,
                proportional_paid=totals[0],
                recurring_paid=totals[1],
                executed_full_l1_notional=totals[2],
            )
        )
    return tuple(result)


class PortfolioSnapshotReference(PortfolioContract):
    """Identify actual complete canonical state and its acknowledged manifest.

    Attributes
    ----------
    component : StateBlobRef
        Actual selected manifest entry: complete owner/codec binding, raw
        canonical state SHA-256 and exact positive UTF-8 byte count.
    manifest_fingerprint : str
        Exact canonical current manifest SHA-256, joined to the observed ACK.
    manifest_byte_count : int
        Complete canonical manifest UTF-8 bytes, at most the existing2MiB cap.

    Notes
    -----
    Fingerprints and byte counts name complete existing artifacts. They neither
    authenticate their producer nor grant file/import authority. The trusted
    resolver must verify the actual current ACK, selected codec and every byte.
    """

    component: StateBlobRef
    manifest_fingerprint: Fingerprint
    manifest_byte_count: Annotated[int, Field(strict=True, gt=0, le=2 * 1024**2)]

    @model_validator(mode="after")
    def _capacity(self) -> Self:
        """Retain the existing complete-state16MiB reader ceiling."""
        if self.component.byte_count > 16 * 1024**2:
            raise ValueError("Portfolio state reference exceeds the original16MiB component ceiling")
        return self


class PortfolioCoordinateResultV2(PortfolioContract):
    """Publish terminal outcomes and exact retained state without duplicating it.

    Attributes
    ----------
    source : DefinitionBinding
        Original independently admitted synthetic source and interpretation.
    snapshot : PortfolioSnapshotReference
        Actual selected checkpoint/manifest identities and byte counts; the state codec
        explicitly distinguishes legacy1 and normalized2.
    experiment_fingerprint : str
        Digest of the exact selected state version, distinct from another version
        or its explicit expanded read view, and from the checkpoint
        envelope's encoded-content digest and from producer authentication.
    proof : OwnedCommitProof
        Actual terminal current receipt with original execution/work bindings.
    outcomes : tuple of PortfolioTerminalOutcome
        Every original candidate, in declared order; exact funded-credit totals.

    Notes
    -----
    Schema2 deliberately changes the coordinate wire. Legacy schema1 readers
    remain separate. No reader migrates an old operational store or infers a
    completed task from this pure artifact alone.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    source: DefinitionBinding
    snapshot: PortfolioSnapshotReference
    experiment_fingerprint: Fingerprint
    proof: OwnedCommitProof
    outcomes: Annotated[
        tuple[PortfolioTerminalOutcome, ...], Field(min_length=len(model.MODES), max_length=len(model.MODES))
    ]

    @model_validator(mode="after")
    def _terminal(self) -> Self:
        """Bind a terminal current receipt, complete owner and ordered outcomes."""
        proof = self.proof
        if (
            proof.frontier.position != 16
            or proof.receipt.accepted_head != proof.receipt.current_head
            or self.snapshot.manifest_fingerprint != proof.observed_head.manifest_fingerprint
            or self.snapshot.component.binding.owner != proof.frontier.coordinate
            or tuple(item.name for item in self.outcomes) != model.MODES
        ):
            raise ValueError("Compact portfolio result requires its complete ordered terminal current ACK")
        return self


class PortfolioReportItemV2(PortfolioContract):
    """Bind a compact coordinate publication to its exact canonical file bytes.

    Attributes
    ----------
    coordinate : PortfolioCoordinateResultV2
        Complete compact typed coordinate evidence, including its terminal ACK.
    artifact_profile : str
        Fixed schema2/core-JSON interpretation; no path, URL or import selector.
    artifact_fingerprint : str
        SHA-256 of the entire canonical coordinate-result file, without newline.
    artifact_byte_count : int
        Exact positive UTF-8 file bytes, bounded by the selected1MiB codec.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    coordinate: PortfolioCoordinateResultV2
    artifact_profile: Literal["portfolio-coordinate-result-v2/core-json-v1"]
    artifact_fingerprint: Fingerprint
    artifact_byte_count: Annotated[int, Field(strict=True, gt=0, le=1024**2)]

    @model_validator(mode="after")
    def _artifact(self) -> Self:
        """Require complete already-validated coordinate bytes, never a prefix."""
        content = canonical_json(self.coordinate.model_dump(mode="json")).encode("utf-8")
        if (len(content), "sha256:" + sha256(content).hexdigest()) != (
            self.artifact_byte_count,
            self.artifact_fingerprint,
        ):
            raise ValueError("Portfolio coordinate reference differs from its exact complete canonical artifact")
        return self


class PortfolioReportV2(PortfolioContract):
    """Return the complete compact result while full receipts remain artifacts.

    Attributes
    ----------
    interpretation : {'constructed_mechanism_only'}
        Descriptive synthetic accounting; no efficacy or inference declaration.
    results : tuple of PortfolioReportItemV2
        The sole original authored trajectory and exact complete file identity.

    Notes
    -----
    One authored trajectory supplies fourteen exact terminal book summaries.
    The complete original checkpoint is referenced, never truncated or treated
    as unavailable. Resolving it requires trusted host storage and current ACK
    verification. Neither a content reference nor this report grants execution
    authority or financial efficacy.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    interpretation: Literal["constructed_mechanism_only"]
    results: Annotated[tuple[PortfolioReportItemV2, ...], Field(min_length=1, max_length=1)]
