"""Structural host-owned entry points; no registration, runtime or inheritance."""

from typing import Protocol

# A narrow structural protocol deliberately needs only one public entry point.
# pylint: disable=too-few-public-methods


class Decider[RequestT, OutputT](Protocol):
    """Describe direct decision calls with caller-selected typed boundaries."""

    def decide(self, request: RequestT, /) -> OutputT:
        """Produce a decision using the implementation's explicit state policy.

        Parameters
        ----------
        request : RequestT
            Caller-selected input; no RunSpec or workspace is required.

        Returns
        -------
        OutputT
            Implementation-selected decision/proposal, not an executed action.

        Notes
        -----
        Implementations document state changes and errors. This protocol itself
        performs no I/O, validation, registration or action execution.
        """
        ...


class Updater[FeedbackT, UpdateT](Protocol):
    """Describe a direct update without imposing scheduling or delivery policy."""

    def update(self, feedback: FeedbackT, /) -> UpdateT:
        """Apply caller-supplied feedback under the implementation's contract.

        Parameters
        ----------
        feedback : FeedbackT
            Explicit typed feedback. Caller handles delivery/deduplication.

        Returns
        -------
        UpdateT
            Implementation-selected update result, including None if declared.

        Notes
        -----
        Updates may mutate implementation state and are not implicitly
        idempotent. Implementations specify accepted coverage, errors and effects.
        """
        ...


class Stateful[StateT](Protocol):
    """Describe explicit snapshots whose representation belongs to the host."""

    def export_state(self) -> StateT:
        """Export the implementation's documented state without framework storage.

        Returns
        -------
        StateT
            Caller-owned snapshot representation and compatibility contract.

        Notes
        -----
        Implementations document copying, persistence needs and side effects.
        This protocol does not promise executable checkpoint compatibility.
        """
        ...

    def restore_state(self, state: StateT, /) -> None:
        """Restore an explicitly compatible host-selected snapshot.

        Parameters
        ----------
        state : StateT
            Snapshot subject to the implementation's version/invariant checks.

        Notes
        -----
        May replace implementation state. Implementations document rejected
        versions and atomicity; no migration or storage is supplied here.
        """
        ...


class Solver[ProblemT, ResultT](Protocol):
    """Describe a direct solver that can be used independently from learning."""

    def solve(self, problem: ProblemT, /) -> ResultT:
        """Return a typed proposal for the supplied problem.

        Parameters
        ----------
        problem : ProblemT
            Host-selected problem description.

        Returns
        -------
        ResultT
            Proposal or explicit unavailable/inconclusive result.

        Notes
        -----
        Implementations document hypotheses, errors, complexity and effects.
        This protocol does not imply optimality or execute an external action.
        """
        ...
