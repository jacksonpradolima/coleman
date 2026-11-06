"""Pure support checks and acyclic projection to existing protected contracts."""

from __future__ import annotations

from typing import Literal, TypedDict, cast

from coleman.core.base import CoreContract
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.evaluation.design._wire import wire_size
from coleman.evaluation.design.contracts import ExperimentDesign, InferenceQualification
from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.protected.bounded_inference import BoundedMeanProtocol
from coleman.evaluation.protected.contracts import EvaluationProtocol, FrozenCandidate, FrozenEvaluation, freeze


class UnsupportedEvaluationError(ValueError):
    """Expose a typed unsupported request without fabricating a protected freeze.

    Attributes
    ----------
    qualification : InferenceQualification
        Actual unsupported assumptions and reasons. This is not missing outcome
        data, failed scientific evidence or an efficacy conclusion.
    """

    def __init__(self, qualification: InferenceQualification) -> None:
        """Retain a pure unsupported qualification without invoking any loader.

        Parameters
        ----------
        qualification : InferenceQualification
            Valid unsupported request. Supported qualifications reject here.

        Raises
        ------
        ValueError
            The supplied qualification is not unsupported.
        """
        qualification = InferenceQualification.model_validate(qualification)
        if qualification.status != "unsupported":
            raise ValueError("UnsupportedEvaluationError requires an unsupported qualification")
        self.qualification = qualification
        super().__init__("Unsupported evaluation assumptions: " + ", ".join(qualification.reasons))


class _ProjectedConfiguration(CoreContract):
    """Bind a complete base configuration to an earlier design and selection."""

    profile: Literal["domain-neutral-protected-candidate-v1"]
    base: FrozenCandidate
    design_fingerprint: Fingerprint
    selection_fingerprint: Fingerprint


class _ProtocolFields(TypedDict):
    """Type the common arguments at the existing protocol constructor boundary."""

    protocol_id: str
    holdout_id: str
    metric_name: str
    unit: str
    meaning: str
    direction: Literal["minimize", "maximize"]
    independent_unit: str
    dependence: Literal["independent"]
    normal_differences_assumed: bool
    practical_effect: float
    alpha: float
    minimum_units: int


def aggregate_meaning(design: ExperimentDesign) -> str:
    """Name the primary raw quantity's explicit equal-panel mean semantics.

    Parameters
    ----------
    design : ExperimentDesign
        Valid frozen primary metric and aggregation declaration.

    Returns
    -------
    str
        Raw meaning plus its separately versioned seed/member/unit mean profile.
        Physical units and the declared horizon are not changed.
    """
    primary = design.metric(design.primary_metric)
    return primary.quantity.spec.semantics.meaning + ":" + design.aggregation


def _unsupported(design: ExperimentDesign) -> tuple[str, ...]:
    """Reject unmatched assumptions without fitting or examining outcomes."""
    request = design.inference
    reasons = []
    if request.dependence != "independent":
        reasons.append("dependent_or_unknown_final_units")
    if request.assignment != "complete_paired_units":
        reasons.append("adaptive_partial_inferential_outcomes")
    if request.analysis != "fixed_final":
        reasons.append("unmatched_optional_stopping")
    if any(item.strategy.update_scope == "across_units" for item in design.candidates):
        reasons.append("strategy_updates_across_final_units")
    if design.metric(design.primary_metric).direction == "none":
        reasons.append("primary_metric_has_no_comparison_direction")
    used = {name for item in design.hypotheses for name in (item.candidate_id, item.baseline_id)}
    if any(item.role == "retrospective_oracle" and item.candidate.candidate_id in used for item in design.candidates):
        reasons.append("retrospective_oracle_comparison")
    return tuple(reasons) + _method_reasons(design)


def _method_reasons(design: ExperimentDesign) -> tuple[str, ...]:
    """Keep known target bounds separate from the legacy t assumptions."""
    request = design.inference
    if request.method == "paired_t_holm_bonferroni_ci_v1":
        return () if request.normal_differences_assumed else ("normal_differences_not_assumed",)
    reasons = []
    quantity = design.metric(design.primary_metric).quantity.spec
    if quantity.minimum is None or quantity.maximum is None or request.bounds_basis is None:
        reasons.append("unknown_or_unjustified_target_bounds")
    elif quantity.minimum == quantity.maximum:
        reasons.append("degenerate_bounded_target_range")
    if request.normal_differences_assumed:
        reasons.append("bounded_profile_does_not_declare_normality")
    return tuple(reasons)


def _protocol(design: ExperimentDesign) -> BoundedMeanProtocol | EvaluationProtocol:
    """Construct only an existing supported protocol after assumption checking."""
    request = design.inference
    primary = design.metric(design.primary_metric)
    fields: _ProtocolFields = {
        "protocol_id": fingerprint(["coleman.generic-evaluation.protocol-v1", design.fingerprint]),
        "holdout_id": design.scope.holdout_id,
        "metric_name": primary.name,
        "unit": primary.quantity.spec.semantics.unit,
        "meaning": aggregate_meaning(design),
        "direction": cast(Literal["minimize", "maximize"], primary.direction),
        "independent_unit": request.independent_unit,
        "dependence": "independent",
        "normal_differences_assumed": request.normal_differences_assumed,
        "practical_effect": request.practical_effect.value,
        "alpha": request.alpha,
        "minimum_units": request.minimum_units,
    }
    if request.method == "paired_t_holm_bonferroni_ci_v1":
        return EvaluationProtocol(**fields)
    return BoundedMeanProtocol(
        schema_version=1,
        outcome_lower=cast(float, primary.quantity.spec.minimum),
        outcome_upper=cast(float, primary.quantity.spec.maximum),
        **fields,
    )


def _checked(design: ExperimentDesign, selection: SelectionManifest) -> tuple[ExperimentDesign, SelectionManifest]:
    """Validate complete declarations before checking inference or projection."""
    design = ExperimentDesign.model_validate(design)
    selection = SelectionManifest.model_validate(selection)
    selection.require_design(design)
    return design, selection


def _qualification(design: ExperimentDesign) -> InferenceQualification:
    """Return the selected method or every known unsupported assumption."""
    reasons = _unsupported(design)
    return InferenceQualification(
        schema_version=1,
        status="unsupported" if reasons else "supported",
        protocol=None if reasons else _protocol(design),
        reasons=reasons,
    )


def qualify_inference(design: ExperimentDesign, selection: SelectionManifest) -> InferenceQualification:
    """Check fixed independent-unit inference without reading outcomes or capabilities.

    Parameters
    ----------
    design : ExperimentDesign
        Complete source, units, treatment laws, quantity and inference request.
    selection : SelectionManifest
        Complete development-only selection bound to the exact final family.

    Returns
    -------
    InferenceQualification
        Supported existing bounded/t protocol or explicit unsupported reasons.
        Support is distinct from source cleanliness, unit independence in reality,
        empirical admission, sufficient outcomes and confirmatory eligibility.

    Raises
    ------
    ValueError
        Invalid contracts, mismatched selection/family or declared capacities.

    Notes
    -----
    No analytics import, installation inspection, normality test, source access,
    clock capture or statistical calculation occurs. A supported t request still
    requires the optional analytics capability at the execution boundary.
    """
    design, _ = _checked(design, selection)
    return _qualification(design)


def project_frozen(design: ExperimentDesign, selection: SelectionManifest) -> FrozenEvaluation:
    """Project a supported complete design through the existing protected contract.

    Parameters
    ----------
    design : ExperimentDesign
        Contains BASE configurations; it cannot contain this later projection.
    selection : SelectionManifest
        Earlier complete development selection and immutable design binding.

    Returns
    -------
    FrozenEvaluation
        Actual projected configurations, common source-derived holdout and the
        complete predeclared family. Data, chronology, units and meaning remain
        explicit. No freeze or selection is persisted by this pure operation.

    Raises
    ------
    UnsupportedEvaluationError
        A known request is outside the existing procedures' assumptions.
    ValueError
        Invalid declarations, changed family or exceeded encoded capacity.

    Notes
    -----
    The dependency order is base configuration → design/selection → projected
    configuration. Raw produced records retain the BASE specification binding;
    derived unit records use this projected specification. Imported evidence
    retains original bindings and receives separate execution-boundary exposure.
    """
    design, selection = _checked(design, selection)
    qualification = _qualification(design)
    if qualification.protocol is None:
        raise UnsupportedEvaluationError(qualification)
    design_digest, selection_digest = design.fingerprint, selection.fingerprint
    candidates = tuple(
        FrozenCandidate.from_configuration(
            item.candidate.candidate_id,
            _ProjectedConfiguration(
                schema_version=1,
                profile="domain-neutral-protected-candidate-v1",
                base=item.candidate,
                design_fingerprint=design_digest,
                selection_fingerprint=selection_digest,
            ).model_dump(mode="json"),
        )
        for item in design.candidates
    )
    projected = freeze(
        candidates=candidates,
        data_fingerprint=design.snapshot.data_fingerprint,
        plan=design.comparison_plan,
        protocol=qualification.protocol,
        comparison_family=design.hypotheses,
    )
    if wire_size(projected) + 1 > design.limits.max_freeze_bytes:
        raise ValueError("Actual protected projection exceeds the declared freeze capacity")
    return projected
