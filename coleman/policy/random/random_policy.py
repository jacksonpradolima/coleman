"""Random policy."""

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy


class RandomPolicy(Policy):
    """Random policy that randomly selects from all available actions.

    No consideration is given to which action is apparently best.
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "Random"

    def choose_all(self, agent: Agent):
        """Choose all actions randomly.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be shuffled.

        Returns
        -------
        list of str
            Randomly ordered list of action names.
        """
        actions = agent.actions["Name"].to_numpy()
        _policy_base._rng.shuffle(actions)
        return actions.tolist()
