"""DynamicBandit - a bandit that supports runtime arm management."""

from .base import Bandit


class DynamicBandit(Bandit):
    """A Bandit that allows dynamic management of its arms.

    Extends the base Bandit to support updating the set of arms at runtime.
    """

    def update_arms(self, arms: list[dict]):
        """Update the arms of the bandit.

        Parameters
        ----------
        arms : list of dict
            The new set of arms to replace the current ones.
        """
        self.reset()
        self.add_arms(arms)
