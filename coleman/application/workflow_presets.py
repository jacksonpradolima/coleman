"""Trusted offline preparation of approved original-source workflow presets."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from coleman.application.workflow_registry import snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_etl import export_lrts_parquet, validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt, LRTSETLRecipe
from coleman.corpus.materialization import SourceCache
from coleman.results.domain import fingerprint
from coleman.workflows.contracts import DatasetManifest, WorkflowContract


class RegisteredKaraf(WorkflowContract):
    """Actual approved registration plus its unchanged original-source conversion receipt.

    Parameters
    ----------
    manifest : DatasetManifest
        Exact native snapshot with explicit retrospective qualification limits.
    receipt : LRTSETLReceipt
        Original pinned hashes, row/exclusion conservation and projection/writer
        identity. Success grants neither historical clocks nor empirical admission.
    """

    manifest: DatasetManifest
    receipt: LRTSETLReceipt


def register_karaf(service: WorkflowService, cache: SourceCache, output: Path) -> RegisteredKaraf:
    """Prepare the existing eight-build Karaf preset offline and register verified native bytes.

    Parameters
    ----------
    service : WorkflowService
        Trusted host workspace/principal; source acquisition is never a tool operation.
    cache : SourceCache
        Permitted pre-populated pinned original-source cache. Missing source bytes
        fail; this function never creates a network reader or fetches a URL.
    output : pathlib.Path
        Host-owned ETL output outside the workflow workspace and source cache/Git. Existing recipe limits,
        publication, hashes and receipt validation remain authoritative.

    Returns
    -------
    RegisteredKaraf
        Exact native registration and full retained typed source/projection receipt.

    Raises
    ------
    ValueError, OSError
        Missing/changed source bytes, unsafe storage, conversion/compatibility
        failure, conflicting registration or finite dataset capacity exhaustion.

    Notes
    -----
    Uses the existing source adapter and Parquet ETL unchanged. Retains its
    exclusion counts, assumed source order and post-hoc serial duration model;
    missing original feedback/availability evidence is never synthesized.
    """
    if output.resolve().is_relative_to(service.store.root.resolve()):
        raise ValueError("Source preparation output must stay outside the workflow workspace")
    recipe = LRTSETLRecipe.from_json(
        files("coleman.corpus.resources").joinpath("lrts-karaf-first-eight-v1.json").read_bytes()
    )
    complete = export_lrts_parquet(recipe, cache, ArtifactWriter(output))
    receipt = LRTSETLReceipt.from_json(complete.read_bytes())
    validate_lrts_etl(receipt, cache, complete.parent)
    with TemporaryDirectory(prefix="coleman-workflow-registration-") as temporary:
        staging = Path(temporary)
        manifest = snapshot_history(
            staging,
            complete.parent / "datasets" / recipe.project / "features-engineered.parquet",
            dataset_id="karaf-lrts-eight-v1",
            source_fingerprint=fingerprint(list(receipt.source_datasets)),
            projection_fingerprint=receipt.fingerprint,
            evidence="descriptive_source",
            assumptions=(
                "Pinned original LRTS Karaf reports; assumed metadata order, not original test execution order.",
                "Post-hoc recorded durations; no reviewed predecision scheduling-cost/feedback clocks.",
                f"Original methods={receipt.observed_cases}; runnable={receipt.replay_cases}; "
                f"not-run exclusions={receipt.excluded_not_run}.",
                "Empirical protected admission remains blocked; retrospective descriptive replay only.",
            ),
        )
        service.register(manifest, snapshot_root=staging)
    ArtifactWriter(service.store.root).write_immutable(
        service.store.root / "source-receipts" / f"{receipt.fingerprint[7:]}.json", receipt.to_json()
    )
    return RegisteredKaraf(schema_version=1, manifest=manifest, receipt=receipt)
