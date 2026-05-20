"""Scenario loaders that read build-level test datasets and yield virtual scenarios."""

import os
import warnings
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from ..virtual import VirtualContextScenario, VirtualHCSScenario, VirtualScenario


class ScenarioLoader:
    """Base class for scenario loaders that process test case data from dataset files.

    Parameters
    ----------
    tcfile : str
        Path to the test case dataset file (Parquet preferred, CSV supported).
    sched_time_ratio : float, optional
        Ratio of scheduled time to total build time. Default is 0.5.

    Attributes
    ----------
    name : str
        Name derived from the dataset directory.
    avail_time_ratio : float
        The available time ratio for scheduling.
    current_build : int
        The current build number being processed.
    total_build_duration : float
        The total duration of the current build.
    scenario : VirtualScenario or None
        The current virtual scenario.
    tcdf : polars.DataFrame
        DataFrame containing test case data.
    max_builds : int
        The maximum number of builds in the dataset.
    """

    REQUIRED_COLUMNS = ["Name", "Duration", "CalcPrio", "LastRun", "Verdict"]

    def __init__(self, tcfile: str, sched_time_ratio: float = 0.5) -> None:
        """Initialize the ScenarioLoader.

        Parameters
        ----------
        tcfile : str
            Path to the test case dataset file.
        sched_time_ratio : float, optional
            Ratio of scheduled time to total build time. Default is 0.5.
        """
        self.name = os.path.split(os.path.dirname(tcfile))[1]
        self.avail_time_ratio = sched_time_ratio
        self.current_build = 0
        self._build_index = -1
        self.total_build_duration = 0.0
        self.scenario: VirtualScenario | None = None
        self._current_build_df: pl.DataFrame | None = None

        self._testcases_lazy = self._read_testcases(tcfile)
        self._testcases_schema = self._testcases_lazy.collect_schema()
        build_ids_df = cast(
            pl.DataFrame,
            self._testcases_lazy.select(pl.col("BuildId").unique().sort()).collect(),
        )
        self._build_ids = cast(list[int], build_ids_df.get_column("BuildId").to_list())
        self.max_builds = len(self._build_ids)

        self._builds_map: dict[int, pl.DataFrame] = {}
        self._prefer_eager_build_map = False

        # For small-to-medium datasets, eager partitioning avoids repeated
        # LazyFrame scans per build and is significantly faster in tight loops.
        tcfile_size_mb = Path(tcfile).stat().st_size / (1024 * 1024)
        if tcfile_size_mb <= 128 and self.max_builds <= 512:
            eager_df = cast(pl.DataFrame, self._testcases_lazy.collect())
            for build_part in eager_df.partition_by("BuildId", maintain_order=False, include_key=True):
                build_id = int(build_part["BuildId"][0])
                self._builds_map[build_id] = build_part.sort("Name")
            self._prefer_eager_build_map = True

        # Keep only a small sliding cache of materialized builds to bound memory usage.
        self._build_cache: OrderedDict[int, pl.DataFrame] = OrderedDict()
        self._build_cache_size = 32

    def _read_testcases(self, tcfile: str) -> pl.LazyFrame:
        """Read the test cases from a provided dataset file.

        Parameters
        ----------
        tcfile : str
            Path to the test case file (Parquet preferred, CSV supported).

        Returns
        -------
        polars.LazyFrame
            LazyFrame containing normalized test case data.
        """
        suffix = Path(tcfile).suffix.lower()
        if suffix == ".parquet":
            df = pl.scan_parquet(tcfile)
        elif suffix == ".csv":
            warnings.warn(
                "CSV scenario files are deprecated and will be removed in a future release. "
                "Please migrate to Parquet files.",
                DeprecationWarning,
                stacklevel=2,
            )
            df = pl.scan_csv(tcfile, separator=";", try_parse_dates=True)
        else:
            msg = f"Unsupported scenario file format: {tcfile!r}. Supported formats: .parquet, .csv"
            raise ValueError(msg)

        build_id_expr = pl.col("BuildId").cast(pl.Int64, strict=False)
        invalid_build_id_count = cast(
            int,
            cast(
                pl.DataFrame,
                df.select(build_id_expr.is_null().sum().alias("invalid_build_ids")).collect(),
            ).item(0, "invalid_build_ids"),
        )
        if invalid_build_id_count > 0:
            msg = "Scenario file contains missing/non-numeric BuildId values; please clean the dataset before loading."
            raise ValueError(msg)

        expressions = [
            build_id_expr.alias("BuildId"),
            pl.col("Name").cast(pl.Utf8, strict=False),
            pl.col("Duration").cast(pl.Float64, strict=False).fill_null(0.0),
        ]

        schema_names = df.collect_schema().names()

        if "LastRun" in schema_names:
            expressions.append(pl.col("LastRun").cast(pl.Utf8, strict=False).fill_null(""))

        if "LastResults" in schema_names:
            expressions.append(pl.col("LastResults").cast(pl.Utf8, strict=False).fill_null(""))

        return df.with_columns(expressions)

    def _collect_build(self, build_id: int, columns: list[str] | None = None) -> pl.DataFrame:
        """Collect one build slice lazily and cache a few recent builds.

        The bounded cache prevents repeated scans for nearby builds while keeping
        memory usage predictable for large datasets.
        """
        if self._prefer_eager_build_map:
            df = self._builds_map.get(build_id, pl.DataFrame(schema=self._testcases_schema))
            if columns:
                return df.select(columns)
            return df

        if build_id in self._build_cache:
            self._build_cache.move_to_end(build_id)
            df = self._build_cache[build_id]
        else:
            df = cast(
                pl.DataFrame,
                self._testcases_lazy.filter(pl.col("BuildId") == build_id).sort("Name").collect(),
            )
            self._build_cache[build_id] = df
            if len(self._build_cache) > self._build_cache_size:
                self._build_cache.popitem(last=False)

        if columns:
            return df.select(columns)
        return df

    @property
    def tcdf(self) -> pl.DataFrame:
        """Legacy eager view of scenario data.

        Returns
        -------
        polars.DataFrame
            Materialized DataFrame. Prefer lazy internal access for scalability.
        """
        warnings.warn(
            "The `tcdf` eager DataFrame attribute is deprecated. "
            "Use loader iteration APIs instead (e.g., iterate the loader with `for scenario in loader`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return cast(pl.DataFrame, self._testcases_lazy.collect())

    def get_avail_time_ratio(self) -> float:
        """Return the available time ratio.

        Returns
        -------
        float
            The available time ratio.
        """
        return self.avail_time_ratio

    def last_build(self, build: int) -> None:
        """Set the last build number.

        Parameters
        ----------
        build : int
            The build number to set.
        """
        if build <= 0:
            self._build_index = -1
            self.current_build = 0
            return

        # `build` is a processed-step count (1..N), not a raw BuildId.
        self._build_index = min(build - 1, self.max_builds - 1)
        self.current_build = self._build_ids[self._build_index] if self._build_index >= 0 else 0

    def get(self) -> VirtualScenario | None:
        """Get the next virtual scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as a scenario.

        Returns
        -------
        VirtualScenario or None
            The next scenario, or None if no more builds remain.
        """
        self._build_index += 1

        if self._build_index >= self.max_builds:
            return None

        self.current_build = self._build_ids[self._build_index]

        self._current_build_df = self._collect_build(self.current_build)

        total_build_duration = cast(int | float | Decimal | None, self._current_build_df["Duration"].sum())
        self.total_build_duration = float(total_build_duration or 0.0)
        available_time = self.total_build_duration * self.avail_time_ratio

        self.scenario = VirtualScenario(
            available_time=available_time,
            testcases=self._current_build_df.select(self.REQUIRED_COLUMNS),
            build_id=self.current_build,
            total_build_duration=self.total_build_duration,
        )

        return self.scenario

    def __str__(self) -> str:
        """Return the name of the scenario loader.

        Returns
        -------
        str
            The scenario loader name.
        """
        return self.name

    def __iter__(self):
        """Return the iterator for the scenario loader.

        Returns
        -------
        ScenarioLoader
            The iterator instance.
        """
        return self

    def __next__(self) -> VirtualScenario:
        """Return the next scenario.

        Returns
        -------
        VirtualScenario
            The next virtual scenario.

        Raises
        ------
        StopIteration
            If no more scenarios are available.
        """
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class HCSScenarioLoader(ScenarioLoader):
    """Scenario loader for HCS-specific data.

    Extends the base scenario loader to support variant-specific data.

    Parameters
    ----------
    tcfile : str
        Path to the test case dataset file.
    variantsfile : str
        Path to the variants dataset file.
    sched_time_ratio : float, optional
        Ratio of scheduled time to total build time. Default is 0.5.

    Attributes
    ----------
    variants : polars.DataFrame
        DataFrame containing variant data.
    """

    def __init__(self, tcfile: str, variantsfile: str, sched_time_ratio=0.5) -> None:
        """Initialize the HCSScenarioLoader.

        Parameters
        ----------
        tcfile : str
            Path to the test case dataset file.
        variantsfile : str
            Path to the variants dataset file.
        sched_time_ratio : float, optional
            Ratio of scheduled time to total build time. Default is 0.5.
        """
        super().__init__(tcfile, sched_time_ratio)

        self._variants_lazy = self._read_variants(variantsfile)
        self._variants_schema = self._variants_lazy.collect_schema()
        variants_df = cast(pl.DataFrame, self._variants_lazy.select(pl.col("Variant").unique()).collect())
        self._variant_names = cast(list[str], variants_df.get_column("Variant").to_list())

        self._variants_map: dict[int, pl.DataFrame] = {}
        self._prefer_eager_variants_map = False

        variants_file_size_mb = Path(variantsfile).stat().st_size / (1024 * 1024)
        if variants_file_size_mb <= 64 and self.max_builds <= 512:
            eager_variants_df = cast(pl.DataFrame, self._variants_lazy.collect())
            for variant_part in eager_variants_df.partition_by("BuildId", maintain_order=False, include_key=True):
                build_id = int(variant_part["BuildId"][0])
                self._variants_map[build_id] = variant_part
            self._prefer_eager_variants_map = True

        self._variants_cache: OrderedDict[int, pl.DataFrame] = OrderedDict()
        self._variants_cache_size = 32

    def _read_variants(self, variantsfile: str) -> pl.LazyFrame:
        """Read the variants from a provided dataset file.

        Parameters
        ----------
        variantsfile : str
            Path to the variants dataset file.

        Returns
        -------
        polars.LazyFrame
            LazyFrame containing variant data.
        """
        suffix = Path(variantsfile).suffix.lower()
        if suffix == ".parquet":
            df = pl.scan_parquet(variantsfile)
        elif suffix == ".csv":
            warnings.warn(
                "CSV variants files are deprecated and will be removed in a future release. "
                "Please migrate to Parquet files.",
                DeprecationWarning,
                stacklevel=2,
            )
            df = pl.scan_csv(variantsfile, separator=";", try_parse_dates=True)
        else:
            msg = f"Unsupported variants file format: {variantsfile!r}. Supported formats: .parquet, .csv"
            raise ValueError(msg)

        return df.with_columns([pl.col("Variant").str.replace_all(r"[!#$%^&*()\[\]{};:,.<>?|`~=+]", "_")])

    @property
    def variants(self) -> pl.DataFrame:
        """Legacy eager view of variants data."""
        warnings.warn(
            "The `variants` eager DataFrame attribute is deprecated. Use scenario-loader APIs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cast(pl.DataFrame, self._variants_lazy.collect())

    def get_total_variants(self):
        """Return the number of unique variants.

        Returns
        -------
        int
            The number of unique variants.
        """
        return len(self._variant_names)

    def get_all_variants(self):
        """Return all unique variant names as a list.

        Returns
        -------
        list of str
            List of unique variant names.
        """
        return list(self._variant_names)

    def get(self) -> VirtualHCSScenario | None:
        """Get the next virtual HCS scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as an HCS scenario with variant data.

        Returns
        -------
        VirtualHCSScenario or None
            The next HCS scenario, or None if no more builds remain.
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        if self._prefer_eager_variants_map:
            variants = self._variants_map.get(self.current_build, pl.DataFrame(schema=self._variants_schema))
        elif self.current_build in self._variants_cache:
            self._variants_cache.move_to_end(self.current_build)
            variants = self._variants_cache[self.current_build]
        else:
            variants = cast(
                pl.DataFrame,
                self._variants_lazy.filter(pl.col("BuildId") == self.current_build).collect(),
            )
            self._variants_cache[self.current_build] = variants
            if len(self._variants_cache) > self._variants_cache_size:
                self._variants_cache.popitem(last=False)

        if variants.height == 0:
            variants = pl.DataFrame(schema=self._variants_schema)

        self.scenario = VirtualHCSScenario(
            available_time=base_scenario.available_time,
            testcases=base_scenario.get_testcases_df(),
            build_id=base_scenario.build_id,
            total_build_duration=base_scenario.total_build_duration,
            variants=variants,
        )

        return self.scenario

    def __next__(self) -> VirtualHCSScenario:
        """Return the next HCS scenario."""
        sc = self.get()
        if sc is None:
            raise StopIteration()
        return sc


class ContextScenarioLoader(ScenarioLoader):
    """Scenario loader for context-aware data.

    Parameters
    ----------
    tcfile : str
        Path to the test case file.
    feature_group_name : str
        Name of the feature group.
    feature_group_values : list of str
        List of features.
    previous_build : list of str
        List of features from the previous build.
    sched_time_ratio : float, optional
        Ratio of the total build time to be used for scheduling. Default is 0.5.

    Attributes
    ----------
    feature_group : str
        The feature group name.
    features : list of str
        List of feature column names.
    previous_build : list of str
        List of features from the previous build.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        tcfile: str,
        feature_group_name: str,
        feature_group_values: list[str],
        previous_build: list[str],
        sched_time_ratio: float = 0.5,
    ):
        """Initialize the context-aware scenario loader.

        Parameters
        ----------
        tcfile : str
            Path to the test case file.
        feature_group_name : str
            Name of the feature group.
        feature_group_values : list of str
            List of features.
        previous_build : list of str
            List of features from the previous build.
        sched_time_ratio : float, optional
            Ratio of the total build time to be used for scheduling.
            Default is 0.5.
        """
        super().__init__(tcfile, sched_time_ratio)
        self.feature_group = feature_group_name
        self.features = feature_group_values
        self.previous_build = previous_build

    def __str__(self):
        """Return the name of the scenario loader.

        Returns
        -------
        str
            The scenario loader name.
        """
        return self.name

    def _merge_context_features(self, build_df):
        """Merge context features from current and previous builds.

        Parameters
        ----------
        build_df : polars.DataFrame
            DataFrame for the current build.

        Returns
        -------
        polars.DataFrame
            DataFrame with merged context features.
        """
        if self._build_index == 0:
            return self._initialize_first_build_features(build_df)

        current_features = set(self.features).difference(self.previous_build)
        previous_features = list(set(self.previous_build).intersection(self.features))

        previous_build_id = self._build_ids[self._build_index - 1]
        previous_build_df = self._collect_build(previous_build_id)

        merged_df = build_df.select(["Name"] + list(current_features))

        if previous_features:
            previous_data = previous_build_df.select(["Name"] + list(previous_features))
            merged_df = merged_df.join(previous_data, on="Name", how="left")

        if previous_features:
            feature_means_df = previous_build_df.select(previous_features).mean()
            fill_exprs = []
            for feature in previous_features:
                mean_val = feature_means_df[feature][0] if feature_means_df.height > 0 else 0.0
                fill_exprs.append(pl.col(feature).fill_null(mean_val).alias(feature))
            if fill_exprs:
                merged_df = merged_df.with_columns(fill_exprs)

        return merged_df

    def _initialize_first_build_features(self, build_df: pl.DataFrame) -> pl.DataFrame:
        """Initialize default feature values for the first build.

        Creates a DataFrame with default values for all features, as the first
        build has no prior context. Each feature is assigned a default value of 1.

        Parameters
        ----------
        build_df : polars.DataFrame
            DataFrame for the current build.

        Returns
        -------
        polars.DataFrame
            DataFrame with default feature values for the first build.
        """
        result = build_df.select(["Name"])
        for feature in self.features:
            result = result.with_columns([pl.lit(1).alias(feature)])
        return result

    def get(self) -> VirtualContextScenario | None:
        """Get the next virtual context scenario.

        Called by ``__next__``. Separates data by builds and returns each
        successive build as a context scenario with feature data.

        Returns
        -------
        VirtualContextScenario or None
            The next context scenario, or None if no more builds remain.
        """
        base_scenario = super().get()

        if not base_scenario:
            return None

        context_features = self._merge_context_features(self._current_build_df)

        self.scenario = VirtualContextScenario(
            available_time=base_scenario.available_time,
            testcases=base_scenario.get_testcases_df(),
            build_id=base_scenario.build_id,
            total_build_duration=base_scenario.total_build_duration,
            feature_group=self.feature_group,
            features=self.features,
            context_features=context_features,
        )

        return self.scenario

    def __next__(self) -> VirtualContextScenario:
        """Return the next contextual scenario."""
        sc = self.get()
        if sc is None:
            raise StopIteration()
        return sc


# Backward-compatible aliases for the old IndustrialDataset* names
IndustrialDatasetScenarioProvider = ScenarioLoader
IndustrialDatasetHCSScenarioProvider = HCSScenarioLoader
IndustrialDatasetContextScenarioProvider = ContextScenarioLoader
