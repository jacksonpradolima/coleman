"""Pure completion manifests for the separately versioned durable task family."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks._base import TaskContract
from coleman.tasks.contracts import EncodedValue, TaskPlan


class TaskCompletion(TaskContract):
    """Bind a complete concrete result without inventing a universal objective.

    Attributes
    ----------
    plan_fingerprint : str
        Full content identity of the retained, resolved task plan.
    source, implementation : DefinitionBinding
        Exact source and captured executor image selected by that plan.
    stage : Reference
        Original logical stage, without conversion to native build stages.
    coordinates : tuple of str
        Complete ordered coordinate content identities; neither repetitions nor
        physical attempts establish independent scientific observations.
    result : EncodedValue
        Complete final result under the executor's selected concrete codec.
        Units, missingness and inference remain responsibilities of that codec.

    Notes
    -----
    This pure manifest performs no I/O, provider discovery or authenticity check.
    A manifest alone does not establish completed lifecycle state: the common
    driver verifies publications and commits its digest in the metadata store.
    Physical attempt declarations and paid costs are separate evidence. The
    model adds no fee, score, probability or efficacy interpretation.
    """

    artifact_type: Literal["completion"]
    plan_fingerprint: Fingerprint
    source: DefinitionBinding
    implementation: DefinitionBinding
    stage: Reference
    coordinates: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=128)]
    result: EncodedValue

    @model_validator(mode="after")
    def _unique(self) -> Self:
        """Refuse duplicate coordinate identities in a supposedly complete set."""
        if len(self.coordinates) != len(set(self.coordinates)):
            raise ValueError("Task completion requires unique ordered coordinate identities")
        return self

    def validate_for(self, plan: TaskPlan) -> None:
        """Check retained interpretation and membership without loading providers.

        Parameters
        ----------
        plan : TaskPlan
            Complete revalidated owned plan, rather than a loose dictionary.

        Raises
        ------
        ValueError
            Changed source, image, stage, coordinate order, codec or plan digest.

        Notes
        -----
        Checks declared joins only. Concrete result validation additionally
        needs the current explicitly approved executor and its selected codec.
        """
        current = type(self).model_validate(self)
        retained = TaskPlan.model_validate(plan)
        actual = (
            current.plan_fingerprint,
            current.source,
            current.implementation,
            current.stage,
            current.coordinates,
            current.result.codec,
        )
        expected = (
            retained.fingerprint,
            retained.source,
            retained.implementation.binding,
            retained.stage,
            tuple(item.fingerprint for item in retained.coordinates),
            retained.implementation.recipe.codec_for("final_result"),
        )
        if actual != expected:
            raise ValueError("Task completion differs from its complete retained plan")
