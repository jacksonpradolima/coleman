"""Bind the pure terminal allocator to COLEMAN's existing acknowledged store.

No provider is called inside a store transition. The host must first persist the
pending request, execute/reconcile it with its producer, then commit the returned
receipt. An uncertain external effect is never automatically retried here.
"""

import hashlib
from functools import partial
from importlib.resources import files

from coleman.acquisition.contracts import AllocationPlan, AllocationState
from coleman.acquisition.session import TerminalAllocator
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.components import ComponentBinding, ComponentCodec
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import (
    CollectionDefinition,
    CommitIntent,
    CommitRequest,
    ComponentSpec,
    ExecutionCompatibility,
)


def implementation() -> DefinitionBinding:
    """Bind the actual installed mathematical/state-owner source bytes.

    Returns
    -------
    DefinitionBinding
        Source-content identity, without authentication or executable migration.

    Raises
    ------
    OSError
        A selected installed source file cannot be read.
    """
    hashes = {
        name: hashlib.sha256(files("coleman.acquisition").joinpath(name).read_bytes()).hexdigest()
        for name in ("model.py", "contracts.py", "_replay.py", "engine.py", "session.py")
    }
    return DefinitionBinding(
        reference=Reference(id="terminal-allocation-owned-implementation", version="1"),
        definition_fingerprint=fingerprint(hashes),
    )


def no_frontier(_state: AllocationState) -> None:
    """Declare publication-only state; no replay causal frontier is inferred.

    Parameters
    ----------
    _state : AllocationState
        Exact concrete snapshot, with no replay-delivery ownership claimed.

    Returns
    -------
    None
        Publication-only profile; this inspector has no side effects.
    """
    return None


def layout(plan: AllocationPlan):
    """Supply one exact concrete codec and fresh allocator to the existing store.

    Parameters
    ----------
    plan : AllocationPlan
        Frozen run identity, complete pool and selected resource profile.

    Returns
    -------
    tuple
        Existing collection definition and one concrete fresh-owner binding.

    Raises
    ------
    ValueError, OSError
        Inconsistent state binding or unreadable installed source.

    Notes
    -----
    This declares ownership but creates no store, publishes no state and calls
    no external provider. Each independent run needs its own plan reference/root.
    """
    binding = StateBinding(
        owner=plan.reference,
        implementation=implementation(),
        state_codec=Reference(id="terminal-allocation-ledger", version="1"),
        configuration_fingerprint=plan.fingerprint,
    )
    codec = ComponentCodec(binding, AllocationState)
    owner = ComponentBinding(codec, partial(TerminalAllocator, plan), no_frontier)
    definition = CollectionDefinition(
        schema_version=1,
        reference=Reference(id="terminal-allocation-collection", version="1"),
        session=plan.reference,
        source_epoch=Reference(id="admitted-pool", version=plan.source_fingerprint),
        children=(),
        components=(
            ComponentSpec(binding=binding, child=None, roles=("algorithm", "domain"), frontier_profile="none"),
        ),
    )
    return definition, (owner,)


def compatibility() -> ExecutionCompatibility:
    """Capture actual installed runtime without an inferred or synthetic identity.

    Returns
    -------
    ExecutionCompatibility
        Base-only selected runtime plus actual source-bound owner definition.

    Raises
    ------
    ValueError, OSError
        Required installed distribution/source identity cannot be captured.

    Notes
    -----
    Reads installed metadata/source bytes; it opens no state store or provider.
    """
    return ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(implementation(),),
    )


def command(generation, expected, content, *, initialize=False):
    """Bind a publication command to its exact predecessor and original receipt.

    Parameters
    ----------
    generation : int
        Consecutive accepted publication ordinal; zero initializes virgin state.
    expected : AcknowledgedHead or None
        Exact accepted predecessor, null only for virgin initialization.
    content : str
        Canonical input-state or original receipt SHA-256 fingerprint.
    initialize : bool, optional
        Explicit initialization, default False for a normal transition.

    Returns
    -------
    CommitRequest
        Typed command identity; no callback, execution or acknowledgement occurs.

    Raises
    ------
    ValueError
        Invalid ordinal, fingerprint or predecessor representation.
    """
    return CommitRequest(
        schema_version=1,
        expected=expected,
        intent=CommitIntent(
            schema_version=1,
            generation=generation,
            kind="initialize" if initialize else "transition",
            content_fingerprint=content,
        ),
    )
