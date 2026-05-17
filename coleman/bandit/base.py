"""Base classes and schema for multi-armed bandits."""

from abc import ABC, abstractmethod

import polars as pl

#: Schema for the bandit arms DataFrame.
#: Columns follow the standard test-case fieldnames used throughout the framework.
BANDIT_SCHEMA: dict = {
    "Name": pl.Utf8,
    "Duration": pl.Float32,
    "CalcPrio": pl.Int32,
    "LastRun": pl.Utf8,
    "NumRan": pl.Int64,
    "NumErrors": pl.Int64,
    "Verdict": pl.Int8,
    "LastResults": pl.Utf8,
}


class Bandit(ABC):
    """Represents a multi-armed bandit model.

    Parameters
    ----------
    arms : list of dict
        The arms of the bandit (test case records). Required columns are
        defined by ``tc_fieldnames``.

    Attributes
    ----------
    tc_fieldnames : list of str
        Column names expected in the arms data.
    arms : polars.DataFrame or None
        DataFrame containing the current arms of the bandit.
    """

    def __init__(self, arms: list[dict]):
        """Initialize a Bandit with its arms.

        Parameters
        ----------
        arms : list of dict
            The arms of the bandit (test case records). Required columns are
            defined by ``tc_fieldnames``.
        """
        self.tc_fieldnames = [
            "Name",
            "Duration",
            "CalcPrio",
            "LastRun",
            "NumRan",
            "NumErrors",
            "Verdict",
            "LastResults",
        ]

        self.arms: pl.DataFrame = pl.DataFrame(schema=BANDIT_SCHEMA)
        self._arm_names: list[str] = []
        self.reset()
        self.add_arms(arms)

    def reset(self):
        """Reset the arms to an empty DataFrame."""
        self.arms = pl.DataFrame(schema=BANDIT_SCHEMA)
        self._arm_names = []

    def get_arms(self) -> list[str]:
        """Retrieve the list of arm names currently associated with the bandit.

        Returns
        -------
        list of str
            List of arm names (test cases).
        """
        return list(self._arm_names)

    def add_arms(self, arms: list[dict]):
        """Add one or multiple arms to the bandit.

        Parameters
        ----------
        arms : list of dict
            List of arms to add.
        """
        if arms:
            last_results = [
                str(arm.get("LastResults"))
                if isinstance(arm.get("LastResults"), list)
                else (arm.get("LastResults") or "")
                for arm in arms
            ]
            data = {field: [arm.get(field) for arm in arms] for field in BANDIT_SCHEMA if field != "LastResults"}
            data["LastResults"] = last_results
            new_arms = pl.DataFrame(data, schema=BANDIT_SCHEMA)
            self.arms = pl.concat([self.arms, new_arms], how="vertical")
            new_names = new_arms["Name"].to_list()
            self._arm_names.extend(new_names)

    @abstractmethod
    def pull(self, action):
        """Simulate pulling an arm.

        To be implemented by subclasses.

        Parameters
        ----------
        action : list
            The action (prioritized test suite) to pull.
        """
        raise NotImplementedError("You must implement this function")

    def update_priority(self, action):
        """Update the Priority column with the priorities.

        Parameters
        ----------
        action : list of str
            List of test cases in order of prioritization.
        """
        if not action:
            self.arms = self.arms.with_columns(pl.lit(0, dtype=pl.Int32).alias("CalcPrio"))
            return

        prio_df = pl.DataFrame(
            {
                "Name": action,
                "CalcPrio": pl.Series(list(range(1, len(action) + 1)), dtype=pl.Int32),
            }
        )
        self.arms = (
            self.arms.drop("CalcPrio")
            .join(prio_df, on="Name", how="left")
            .with_columns(pl.col("CalcPrio").fill_null(0).cast(pl.Int32))
        )
