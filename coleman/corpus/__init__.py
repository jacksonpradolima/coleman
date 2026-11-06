"""Versioned CI corpus metadata and bounded synthetic correctness validation.

Phase-one source registration is not the complete diversified benchmark program.
Source admission, licensing and generalization remain separate evidence claims.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.corpus.contracts import (
        AdapterReference,
        ConversionSummary,
        CorpusManifest,
        CorpusResources,
        DataCard,
        EvaluationDeclaration,
        ExclusionCount,
        LicenseReview,
        ObservationIdentity,
        OutcomeSemantics,
        SourceArtifact,
        TimeSemantics,
        VerdictMapping,
    )
    from coleman.corpus.fixtures import (
        CIReferenceFixture,
        ParallelFixtureCase,
        ParallelReference,
        SerialFixtureCase,
        fixture_bytes,
        parallel_reference,
    )
    from coleman.corpus.materialization import (
        SourceCache,
        materialize_source,
        publish_materialization,
        validate_cache_separation,
        validate_materialization,
        validate_source_storage,
    )
    from coleman.corpus.materialization_contracts import (
        MaterializationLimits,
        MaterializationReceipt,
        MaterializationSpec,
        PackageMember,
    )
    from coleman.corpus.profiles import (
        ColumnMissingness,
        CorpusProfileInventory,
        NativeSourceProfile,
        NumericProfile,
        profile_native_source,
        publish_profile_inventory,
        publish_source_profile,
    )
    from coleman.corpus.reference import RatioReference, TinyCIReference, tiny_ci_reference
    from coleman.corpus.registry import (
        REQUIRED_REGIMES,
        CorpusRegistry,
        FixtureReadiness,
        bundled_fixture_bytes,
        fixture_readiness,
        load_registry,
        publish_fixture_readiness,
        validate_bundled_source,
    )
    from coleman.corpus.validation import CorpusFixtureReport, publish_fixture_report, validate_fixture_source

_EXPORTS = {
    "AdapterReference": ("coleman.corpus.contracts", "AdapterReference"),
    "CIReferenceFixture": ("coleman.corpus.fixtures", "CIReferenceFixture"),
    "ColumnMissingness": ("coleman.corpus.profiles", "ColumnMissingness"),
    "ConversionSummary": ("coleman.corpus.contracts", "ConversionSummary"),
    "CorpusFixtureReport": ("coleman.corpus.validation", "CorpusFixtureReport"),
    "CorpusManifest": ("coleman.corpus.contracts", "CorpusManifest"),
    "CorpusProfileInventory": ("coleman.corpus.profiles", "CorpusProfileInventory"),
    "CorpusRegistry": ("coleman.corpus.registry", "CorpusRegistry"),
    "CorpusResources": ("coleman.corpus.contracts", "CorpusResources"),
    "DataCard": ("coleman.corpus.contracts", "DataCard"),
    "EvaluationDeclaration": ("coleman.corpus.contracts", "EvaluationDeclaration"),
    "ExclusionCount": ("coleman.corpus.contracts", "ExclusionCount"),
    "FixtureReadiness": ("coleman.corpus.registry", "FixtureReadiness"),
    "LicenseReview": ("coleman.corpus.contracts", "LicenseReview"),
    "MaterializationLimits": ("coleman.corpus.materialization_contracts", "MaterializationLimits"),
    "MaterializationReceipt": ("coleman.corpus.materialization_contracts", "MaterializationReceipt"),
    "MaterializationSpec": ("coleman.corpus.materialization_contracts", "MaterializationSpec"),
    "NativeSourceProfile": ("coleman.corpus.profiles", "NativeSourceProfile"),
    "NumericProfile": ("coleman.corpus.profiles", "NumericProfile"),
    "ObservationIdentity": ("coleman.corpus.contracts", "ObservationIdentity"),
    "OutcomeSemantics": ("coleman.corpus.contracts", "OutcomeSemantics"),
    "PackageMember": ("coleman.corpus.materialization_contracts", "PackageMember"),
    "ParallelFixtureCase": ("coleman.corpus.fixtures", "ParallelFixtureCase"),
    "ParallelReference": ("coleman.corpus.fixtures", "ParallelReference"),
    "REQUIRED_REGIMES": ("coleman.corpus.registry", "REQUIRED_REGIMES"),
    "RatioReference": ("coleman.corpus.reference", "RatioReference"),
    "SerialFixtureCase": ("coleman.corpus.fixtures", "SerialFixtureCase"),
    "SourceArtifact": ("coleman.corpus.contracts", "SourceArtifact"),
    "SourceCache": ("coleman.corpus.materialization", "SourceCache"),
    "TimeSemantics": ("coleman.corpus.contracts", "TimeSemantics"),
    "TinyCIReference": ("coleman.corpus.reference", "TinyCIReference"),
    "VerdictMapping": ("coleman.corpus.contracts", "VerdictMapping"),
    "bundled_fixture_bytes": ("coleman.corpus.registry", "bundled_fixture_bytes"),
    "fixture_bytes": ("coleman.corpus.fixtures", "fixture_bytes"),
    "fixture_readiness": ("coleman.corpus.registry", "fixture_readiness"),
    "load_registry": ("coleman.corpus.registry", "load_registry"),
    "materialize_source": ("coleman.corpus.materialization", "materialize_source"),
    "parallel_reference": ("coleman.corpus.fixtures", "parallel_reference"),
    "profile_native_source": ("coleman.corpus.profiles", "profile_native_source"),
    "publish_fixture_readiness": ("coleman.corpus.registry", "publish_fixture_readiness"),
    "publish_fixture_report": ("coleman.corpus.validation", "publish_fixture_report"),
    "publish_materialization": ("coleman.corpus.materialization", "publish_materialization"),
    "publish_profile_inventory": ("coleman.corpus.profiles", "publish_profile_inventory"),
    "publish_source_profile": ("coleman.corpus.profiles", "publish_source_profile"),
    "tiny_ci_reference": ("coleman.corpus.reference", "tiny_ci_reference"),
    "validate_bundled_source": ("coleman.corpus.registry", "validate_bundled_source"),
    "validate_fixture_source": ("coleman.corpus.validation", "validate_fixture_source"),
    "validate_cache_separation": ("coleman.corpus.materialization", "validate_cache_separation"),
    "validate_materialization": ("coleman.corpus.materialization", "validate_materialization"),
    "validate_source_storage": ("coleman.corpus.materialization", "validate_source_storage"),
}

__all__ = [
    "AdapterReference",
    "CIReferenceFixture",
    "ColumnMissingness",
    "ConversionSummary",
    "CorpusFixtureReport",
    "CorpusManifest",
    "CorpusProfileInventory",
    "CorpusRegistry",
    "CorpusResources",
    "DataCard",
    "EvaluationDeclaration",
    "ExclusionCount",
    "FixtureReadiness",
    "LicenseReview",
    "MaterializationLimits",
    "MaterializationReceipt",
    "MaterializationSpec",
    "NativeSourceProfile",
    "NumericProfile",
    "ObservationIdentity",
    "OutcomeSemantics",
    "PackageMember",
    "ParallelFixtureCase",
    "ParallelReference",
    "REQUIRED_REGIMES",
    "RatioReference",
    "SerialFixtureCase",
    "SourceArtifact",
    "SourceCache",
    "TimeSemantics",
    "TinyCIReference",
    "VerdictMapping",
    "bundled_fixture_bytes",
    "fixture_bytes",
    "fixture_readiness",
    "load_registry",
    "materialize_source",
    "parallel_reference",
    "profile_native_source",
    "publish_fixture_readiness",
    "publish_fixture_report",
    "publish_materialization",
    "publish_profile_inventory",
    "publish_source_profile",
    "tiny_ci_reference",
    "validate_bundled_source",
    "validate_fixture_source",
    "validate_cache_separation",
    "validate_materialization",
    "validate_source_storage",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
