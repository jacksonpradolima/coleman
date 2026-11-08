"""Statically declare concrete codecs, without capture or executor imports."""

from __future__ import annotations

from dataclasses import dataclass

from coleman.core._wire import guard_graph
from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.tasks.contracts import OwnedCommitProof
from coleman.tasks.identity import (
    CodecDeclaration,
    CodecRole,
    CodecSelection,
    DistributionRoot,
    EntryPointSelection,
    ExecutorRecipe,
)

from .contracts import (
    HedgeCheckpoint,
    HedgeConfiguration,
    HedgeCoordinateResult,
    HedgeInputs,
    HedgeReport,
    HedgeTrajectory,
)


def reference(name: str) -> Reference:
    """Declare an explicit version-one fixture identity without discovery.

    Parameters
    ----------
    name : str
        Caller-selected stable logical identifier.

    Returns
    -------
    Reference
        Immutable local identity, with no import authority.
    """
    return Reference(id=name, version="1")


def binding(name: str, content: object) -> DefinitionBinding:
    """Bind declared finite semantic content without accessing installation.

    Parameters
    ----------
    name : str
        Explicit version-one definition identifier.
    content : object
        Canonical JSON-compatible complete semantic definition.

    Returns
    -------
    DefinitionBinding
        Content identity, not authenticated code approval.
    """
    return DefinitionBinding(reference=reference(name), definition_fingerprint=fingerprint(content))


def frontier_definition() -> DefinitionBinding:
    """Return accepted original full-panel work semantics, independent of ACKs.

    Returns
    -------
    DefinitionBinding
        Position equals complete inner Hedge U. Publication may batch two panels.
    """
    return binding("hedge-accepted-panel-frontier", {"unit": "accepted-full-panel", "profile": "inner-U-v1"})


@dataclass(frozen=True)
class ModelCodec[T: CoreContract]:
    """Select one concrete bounded model; no encoded text selects Python code.

    Parameters
    ----------
    model : type of CoreContract
        Exact complete concrete model, selected by trusted fixture source.
    maximum_bytes : int
        UTF-8 body capacity, also applied before decoding/revalidation.
    """

    model: type[T]
    maximum_bytes: int

    @property
    def declaration(self) -> CodecDeclaration:
        """Return the exact selected schema, codec law and byte capacity."""
        schema = fingerprint(self.model.model_json_schema())
        return CodecDeclaration(
            binding=binding("typed-hedge-" + self.model.__name__, {"schema": schema, "canonical": "core-json-v1"}),
            schema_fingerprint=schema,
            maximum_bytes=self.maximum_bytes,
        )

    def validate(self, value: object, /) -> T:
        """Revalidate the complete exact selected type within a graph bound.

        Parameters
        ----------
        value : object
            Complete candidate, including an unchecked typed model.

        Returns
        -------
        CoreContract
            Exact concrete validated model; no subclass field truncation.

        Raises
        ------
        TypeError, ValueError
            Wrong concrete type, invalid fields or excessive/deep graph.
        """
        if type(value) not in (self.model,):
            raise TypeError("Fixture codec requires its exact complete concrete model")
        guard_graph(value, maximum_bytes=self.maximum_bytes, maximum_items=100_000)
        return self.model.model_validate(value)

    def decode(self, body: str, /) -> T:
        """Read bounded canonical JSON using only the selected model.

        Parameters
        ----------
        body : str
            Complete UTF-8 JSON, never a path/import expression.

        Returns
        -------
        CoreContract
            Exact complete concrete value, with unknown versions/fields rejected.

        Raises
        ------
        ValueError, TypeError
            Invalid, oversized, deep or noncanonical content.
        """
        value = self.model.from_json(bounded_json_bytes(body, self.maximum_bytes))
        if self.encode(value) != body:
            raise ValueError("Fixture codec requires exact canonical complete bytes")
        return value

    def encode(self, value: CoreContract, /) -> str:
        """Encode revalidated complete fields without dropping optional content.

        Parameters
        ----------
        value : CoreContract
            Actual selected concrete value.

        Returns
        -------
        str
            Complete canonical UTF-8 body within the selected byte cap.
        """
        result = self.validate(value).to_json()
        bounded_json_bytes(result, self.maximum_bytes)
        return result


CONFIGURATION = ModelCodec(HedgeConfiguration, 64 * 1024)
INPUTS = ModelCodec(HedgeInputs, 1024 * 1024)
COORDINATE = ModelCodec(HedgeTrajectory, 128 * 1024)
STATE = ModelCodec(HedgeCheckpoint, 128 * 1024)
COORDINATE_RESULT = ModelCodec(HedgeCoordinateResult, 256 * 1024)
FINAL_RESULT = ModelCodec(HedgeReport, 2 * 1024**2)
PROGRESS_PROOF = ModelCodec(OwnedCommitProof, 128 * 1024)


def recipe() -> ExecutorRecipe:
    """Declare all capabilities without importing provider execution or capture.

    Returns
    -------
    ExecutorRecipe
        Static approved recipe candidate, with five value codecs and state/proof
        roles. An operator must separately approve its actual captured identity.
    """
    roles: tuple[tuple[CodecRole, ModelCodec], ...] = (
        ("configuration", CONFIGURATION),
        ("inputs", INPUTS),
        ("coordinate", COORDINATE),
        ("coordinate_result", COORDINATE_RESULT),
        ("final_result", FINAL_RESULT),
        ("state", STATE),
        ("progress_proof", PROGRESS_PROOF),
    )
    return ExecutorRecipe(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="executor_recipe",
        reference=reference("external-constant-hedge"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        entry_point=EntryPointSelection(
            distribution="coleman-typed-hedge-reference",
            group="coleman.executors.v1",
            name="hedge-reference",
            module="coleman_typed_hedge",
            attribute="bootstrap",
        ),
        roots=(DistributionRoot(name="coleman"), DistributionRoot(name="coleman-typed-hedge-reference")),
        codecs=tuple(CodecSelection(role=role, codec=codec.declaration) for role, codec in roles),
        frontier=frontier_definition(),
        effects="local_state",
    )
