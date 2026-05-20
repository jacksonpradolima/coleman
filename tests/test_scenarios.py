"""Tests for the monitor utility module."""

from pathlib import Path

import polars as pl
import pytest

from coleman.runner import get_scenario_provider
from coleman.scenarios import (
    ContextScenarioLoader,
    HCSScenarioLoader,
    IndustrialDatasetContextScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
    ScenarioLoader,
    VirtualContextScenario,
    VirtualHCSScenario,
    VirtualScenario,
)

# ------------------------ Fixtures for Mock Data ------------------------


@pytest.fixture
def mock_testcases():
    """Fixture to generate a mock DataFrame of test cases."""
    return [
        {"Name": "TC1", "Duration": 1.0, "CalcPrio": 0, "LastRun": "2023-01-01", "Verdict": 1},
        {"Name": "TC2", "Duration": 2.0, "CalcPrio": 1, "LastRun": "2023-01-02", "Verdict": 0},
        {"Name": "TC3", "Duration": 3.5, "CalcPrio": 2, "LastRun": "2023-01-03", "Verdict": 1},
    ]


@pytest.fixture
def mock_variants():
    """Fixture to generate a mock DataFrame of variants."""
    return pl.DataFrame(
        {"BuildId": [1, 1, 2], "Variant": ["A", "B", "A"], "LastRun": ["2023-01-01", "2023-01-02", "2023-01-03"]}
    )


@pytest.fixture
def mock_csv_data(mock_testcases):
    """Fixture to generate a mock CSV DataFrame."""
    data = pl.DataFrame(mock_testcases)
    data = data.with_columns(pl.Series("BuildId", [1, 1, 2]))
    return data


# ------------------------ Fixtures for Large Mock Data ------------------------


@pytest.fixture
def large_testcases():
    """Fixture to generate a large number of test cases."""
    n = 10000  # Large number of test cases
    return [
        {"Name": f"TC{i}", "Duration": i % 10 + 1, "CalcPrio": i % 5, "LastRun": "2023-01-01", "Verdict": 1}
        for i in range(n)
    ]


@pytest.fixture
def large_variants():
    """Fixture to generate a large dataset of variants."""
    return pl.DataFrame(
        {
            "BuildId": [i % 100 for i in range(10000)],  # Simulate 100 builds
            "Variant": [f"Variant_{i % 5}" for i in range(10000)],
            "LastRun": ["2023-01-01"] * 10000,
        }
    )


@pytest.fixture
def large_csv_data(large_testcases):
    """Fixture to generate a large CSV-like DataFrame."""
    data = pl.DataFrame(large_testcases)
    data = data.with_columns(pl.Series("BuildId", [i % 100 for i in range(len(data))]))  # 100 builds
    return data


# ------------------------ Unit Tests ------------------------


# VirtualScenario
def test_virtual_scenario_initialization(mock_testcases):
    """Test initialization of VirtualScenario."""
    scenario = VirtualScenario(available_time=10, testcases=mock_testcases, build_id=1, total_build_duration=6.5)
    assert scenario.get_available_time() == 10
    assert scenario.get_testcases() == mock_testcases
    assert scenario.build_id == 1
    assert abs(scenario.total_build_duration - 6.5) < 1e-6


def test_virtual_scenario_testcases_setter_with_empty_list():
    """Test setting testcases property with an empty list."""
    scenario = VirtualScenario(available_time=10, testcases=[], build_id=1, total_build_duration=6.5)
    assert scenario.get_testcases() == []
    # Set new empty list via property
    scenario.testcases = []
    assert scenario.get_testcases() == []


def test_virtual_scenario_testcases_setter_with_dataframe(mock_testcases):
    """Test setting testcases property with a DataFrame."""
    df = pl.DataFrame(mock_testcases)
    scenario = VirtualScenario(available_time=10, testcases=[], build_id=1, total_build_duration=6.5)
    scenario.testcases = df
    assert len(scenario.get_testcases()) == len(mock_testcases)


def test_virtual_scenario_reset(mock_testcases):
    """Test `reset` method of VirtualScenario."""
    scenario = VirtualScenario(10, mock_testcases, 1, 6.5)
    scenario.reset()
    for case in scenario.get_testcases():
        assert case["CalcPrio"] == 0, "CalcPrio should be reset to 0."


# VirtualHCSScenario
def test_virtual_hcs_scenario(mock_testcases, mock_variants):
    """Test VirtualHCSScenario attributes and methods."""
    scenario = VirtualHCSScenario(
        available_time=10, testcases=mock_testcases, build_id=1, total_build_duration=6.5, variants=mock_variants
    )
    assert scenario.get_variants().equals(mock_variants), "Variants should match the input DataFrame."


# VirtualContextScenario
def test_virtual_context_scenario(mock_testcases):
    """Test VirtualContextScenario initialization and getters."""
    context_features = pl.DataFrame({"Name": ["TC1", "TC2"], "feat1": [0.5, 1.0], "feat2": [1.0, 0.2]})
    scenario = VirtualContextScenario(
        available_time=10,
        testcases=mock_testcases,
        build_id=1,
        total_build_duration=6.5,
        feature_group="test_group",
        features=["feat1", "feat2"],
        context_features=context_features,
    )
    assert scenario.get_feature_group() == "test_group", "Feature group name should match."
    assert scenario.get_features() == ["feat1", "feat2"], "Features should match the input."
    assert scenario.get_context_features().equals(context_features), "Context features should match."


# IndustrialDatasetScenarioProvider
def test_industrial_dataset_scenario_provider(mock_csv_data, tmp_path):
    """Test IndustrialDatasetScenarioProvider functionality, ensuring available_time is computed correctly."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.write_csv(csv_file, separator=";")

    # Initialize provider with sched_time_ratio = 0.5
    provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)
    scenario = next(provider)  # Retrieve the first scenario

    # Debugging output
    print("Total Build Duration:", scenario.total_build_duration)
    print("Expected Available Time:", 0.5 * scenario.total_build_duration)
    print("Scenario Available Time:", scenario.get_available_time())

    # Verify total duration and available time
    assert abs(scenario.get_available_time() - 0.5 * scenario.total_build_duration) < 1e-6, (
        f"Available time should be half of total duration. "
        f"Expected: {0.5 * scenario.total_build_duration}, Got: {scenario.get_available_time()}"
    )
    assert len(scenario.get_testcases()) == 2, "Should retrieve test cases for BuildId=1 only."


def test_industrial_dataset_scenario_provider_casts_name_to_string(tmp_path):
    """Test that numeric Name values are normalized to strings when loading datasets."""
    csv_file = tmp_path / "testcases_numeric_names.csv"
    pl.DataFrame(
        {
            "Name": [1, 2],
            "Duration": [1.0, 2.0],
            "CalcPrio": [0, 0],
            "LastRun": ["2023-01-01", "2023-01-02"],
            "Verdict": [1, 0],
            "BuildId": [1, 1],
        }
    ).write_csv(csv_file, separator=";")

    provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)
    scenario = next(provider)

    assert scenario.get_testcases()[0]["Name"] == "1"
    assert scenario.get_testcases()[1]["Name"] == "2"


def test_industrial_dataset_scenario_provider_parquet_first(mock_csv_data, tmp_path):
    """Test that providers support Parquet datasets without CSV fallback."""
    parquet_file = tmp_path / "testcases.parquet"
    mock_csv_data.write_parquet(parquet_file)

    provider = IndustrialDatasetScenarioProvider(str(parquet_file), sched_time_ratio=0.5)
    scenario = next(provider)

    assert len(scenario.get_testcases()) == 2


def test_industrial_dataset_scenario_provider_csv_deprecation_warning(mock_csv_data, tmp_path):
    """Test that CSV inputs remain supported with a deprecation warning."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.write_csv(csv_file, separator=";")

    with pytest.warns(DeprecationWarning, match="CSV scenario files are deprecated"):
        provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)
        scenario = next(provider)

    assert len(scenario.get_testcases()) == 2


def test_scenario_loader_unsupported_format_raises(tmp_path):
    """Unsupported tcfile suffix should raise ValueError."""
    bad_file = tmp_path / "testcases.txt"
    bad_file.write_text("not used", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported scenario file format"):
        ScenarioLoader(str(bad_file), sched_time_ratio=0.5)


def test_scenario_loader_rejects_invalid_build_id_values(tmp_path):
    """Loader should fail fast when BuildId values are missing/non-numeric."""
    parquet_file = tmp_path / "invalid_build_id.parquet"
    pl.DataFrame(
        {
            "BuildId": ["1", None, "x"],
            "Name": ["A", "B", "C"],
            "Duration": [1.0, 2.0, 3.0],
            "CalcPrio": [0, 0, 0],
            "LastRun": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Verdict": [1, 0, 1],
        }
    ).write_parquet(parquet_file)

    with pytest.raises(ValueError, match="BuildId"):
        ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)


def test_scenario_loader_collect_build_with_columns(mock_csv_data, tmp_path):
    """Exercise _collect_build projection path with explicit columns."""
    parquet_file = tmp_path / "testcases.parquet"
    mock_csv_data.write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)
    build_df = loader._collect_build(1, columns=["Name", "Duration"])  # pylint: disable=protected-access
    assert build_df.columns == ["Name", "Duration"]


def test_scenario_loader_uses_existing_build_ids_when_sparse(tmp_path):
    """Sparse BuildId datasets should iterate only over real builds."""
    parquet_file = tmp_path / "sparse_builds.parquet"
    pl.DataFrame(
        {
            "BuildId": [10, 10, 20],
            "Name": ["TC1", "TC2", "TC1"],
            "Duration": [1.0, 2.0, 3.0],
            "CalcPrio": [0, 0, 0],
            "LastRun": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "Verdict": [1, 0, 1],
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)
    scenarios = list(loader)

    assert loader.max_builds == 2
    assert [sc.build_id for sc in scenarios] == [10, 20]


def test_scenario_loader_current_build_includes_all_rows_eager(tmp_path):
    """Current build must include all rows for that BuildId in eager mode."""
    parquet_file = tmp_path / "all_rows_eager.parquet"
    pl.DataFrame(
        {
            "BuildId": [1, 1, 1, 2],
            "Name": ["TC1", "TC2", "TC3", "TC4"],
            "Duration": [1.0, 2.0, 3.0, 4.0],
            "CalcPrio": [0, 0, 0, 0],
            "LastRun": ["2023-01-01"] * 4,
            "Verdict": [0, 1, 0, 1],
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)

    assert loader._prefer_eager_build_map  # pylint: disable=protected-access

    first_scenario = next(loader)
    assert first_scenario.build_id == 1
    assert len(first_scenario.get_testcases()) == 3

    build_df = loader._collect_build(1)  # pylint: disable=protected-access
    assert build_df.height == 3
    assert set(build_df["Name"].to_list()) == {"TC1", "TC2", "TC3"}


def test_scenario_loader_current_build_includes_all_rows_lazy(tmp_path):
    """Current build must include all rows for that BuildId in lazy mode."""
    parquet_file = tmp_path / "all_rows_lazy.parquet"

    build_ids = [1, 1, 1, 1] + list(range(2, 514))
    names = [f"TC{i}" for i in range(len(build_ids))]
    durations = [1.0] * len(build_ids)
    calc_prio = [0] * len(build_ids)
    last_run = ["2023-01-01"] * len(build_ids)
    verdict = [0] * len(build_ids)

    pl.DataFrame(
        {
            "BuildId": build_ids,
            "Name": names,
            "Duration": durations,
            "CalcPrio": calc_prio,
            "LastRun": last_run,
            "Verdict": verdict,
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)

    assert not loader._prefer_eager_build_map  # pylint: disable=protected-access

    first_scenario = next(loader)
    assert first_scenario.build_id == 1
    assert len(first_scenario.get_testcases()) == 4

    build_df = loader._collect_build(1)  # pylint: disable=protected-access
    assert build_df.height == 4
    assert build_df["BuildId"].to_list() == [1, 1, 1, 1]


def test_last_build_positive_sets_build_index_and_current_build(tmp_path):
    """last_build(n>0) must set _build_index and current_build to the n-th real BuildId."""
    parquet_file = tmp_path / "lb_positive.parquet"
    pl.DataFrame(
        {
            "BuildId": [5, 5, 15, 25],
            "Name": ["TC1", "TC2", "TC1", "TC1"],
            "Duration": [1.0, 2.0, 3.0, 4.0],
            "CalcPrio": [0, 0, 0, 0],
            "LastRun": ["2023-01-01"] * 4,
            "Verdict": [1, 0, 1, 1],
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)

    # last_build(1) → first real BuildId (5)
    loader.last_build(1)
    assert loader._build_index == 0  # pylint: disable=protected-access
    assert loader.current_build == 5

    # last_build(2) → second real BuildId (15)
    loader.last_build(2)
    assert loader._build_index == 1  # pylint: disable=protected-access
    assert loader.current_build == 15

    # last_build beyond max clamps to last available
    loader.last_build(999)
    assert loader._build_index == 2  # pylint: disable=protected-access
    assert loader.current_build == 25


def test_scenario_loader_build_cache_is_bounded(tmp_path):
    """Build cache must stay bounded to avoid unbounded memory growth."""
    parquet_file = tmp_path / "cache_bounds.parquet"
    pl.DataFrame(
        {
            "BuildId": [1, 2, 3, 4],
            "Name": ["A", "A", "A", "A"],
            "Duration": [1.0, 1.0, 1.0, 1.0],
            "CalcPrio": [0, 0, 0, 0],
            "LastRun": ["2023-01-01"] * 4,
            "Verdict": [0, 0, 0, 0],
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)
    for build_id in [1, 2, 3, 4]:
        _ = loader._collect_build(build_id)  # pylint: disable=protected-access

    assert len(loader._build_cache) <= loader._build_cache_size  # pylint: disable=protected-access


def test_scenario_loader_collect_build_with_columns_filter(tmp_path):
    """Cover _collect_build with columns parameter to select specific columns."""
    parquet_file = tmp_path / "cols.parquet"
    pl.DataFrame(
        {
            "BuildId": [1, 1],
            "Name": ["A", "B"],
            "Duration": [1.0, 2.0],
            "CalcPrio": [0, 0],
            "LastRun": ["2023-01-01", "2023-01-02"],
            "Verdict": [1, 0],
        }
    ).write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)
    result = loader._collect_build(1, columns=["Name", "Duration"])  # pylint: disable=protected-access
    assert set(result.columns) == {"Name", "Duration"}
    assert result.height == 2


def test_scenario_loader_tcdf_property_warns_and_returns_df(mock_csv_data, tmp_path):
    """Cover deprecated tcdf property warning/return path (lines 129-135)."""
    parquet_file = tmp_path / "testcases.parquet"
    mock_csv_data.write_parquet(parquet_file)

    loader = ScenarioLoader(str(parquet_file), sched_time_ratio=0.5)
    with pytest.warns(DeprecationWarning, match="tcdf"):
        eager_df = loader.tcdf
    assert eager_df.height > 0


def test_provider_stop_iteration(mock_csv_data, tmp_path):
    """Test StopIteration after all scenarios are retrieved."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.write_csv(csv_file, separator=";")

    provider = IndustrialDatasetScenarioProvider(str(csv_file))
    scenarios = list(provider)  # Consume all scenarios
    assert len(scenarios) == 2, "Should return scenarios for unique BuildIds."
    with pytest.raises(StopIteration):
        next(provider)


def test_hcs_loader_variants_csv_deprecation_warning(mock_csv_data, mock_variants, tmp_path):
    """CSV variants inputs should emit deprecation warnings."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.csv"
    mock_csv_data.write_parquet(tc_file)
    mock_variants.write_csv(variants_file, separator=";")

    with pytest.warns(DeprecationWarning, match="CSV variants files are deprecated"):
        loader = HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)
        assert loader.get_total_variants() > 0


def test_hcs_loader_variants_unsupported_format_raises(mock_csv_data, tmp_path):
    """Unsupported variants suffix should raise ValueError."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.unsupported"
    mock_csv_data.write_parquet(tc_file)
    variants_file.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported variants file format"):
        HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)


def test_hcs_loader_variants_property_warns(mock_csv_data, mock_variants, tmp_path):
    """Accessing legacy variants property emits deprecation warning."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.parquet"
    mock_csv_data.write_parquet(tc_file)
    mock_variants.write_parquet(variants_file)

    loader = HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)
    with pytest.warns(DeprecationWarning, match="variants"):
        variants_df = loader.variants
    assert variants_df.height > 0


def test_hcs_loader_next_raises_stop_iteration_when_exhausted(mock_csv_data, mock_variants, tmp_path):
    """Cover HCS __next__ StopIteration branch (line 327)."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.parquet"
    mock_csv_data.write_parquet(tc_file)
    mock_variants.write_parquet(variants_file)

    loader = HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)
    list(loader)
    with pytest.raises(StopIteration):
        next(loader)


def test_hcs_loader_get_all_variants_returns_list(mock_csv_data, mock_variants, tmp_path):
    """Cover get_all_variants return path (line 327)."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.parquet"
    mock_csv_data.write_parquet(tc_file)
    mock_variants.write_parquet(variants_file)

    loader = HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)
    variants = loader.get_all_variants()
    assert isinstance(variants, list)
    assert len(variants) > 0


def test_context_loader_str_returns_dataset_name(tmp_path):
    """Cover ContextScenarioLoader.__str__ return path (line 422)."""
    tc_file = tmp_path / "ctx.parquet"
    pl.DataFrame(
        {
            "BuildId": [1],
            "Name": ["A"],
            "Duration": [1.0],
            "CalcPrio": [0],
            "LastRun": ["2023-01-01"],
            "Verdict": [1],
            "LastResults": ["P"],
        }
    ).write_parquet(tc_file)

    loader = IndustrialDatasetContextScenarioProvider(
        str(tc_file),
        feature_group_name="ctx",
        feature_group_values=["Duration"],
        previous_build=["Duration"],
        sched_time_ratio=0.5,
    )
    assert str(loader) == tc_file.parent.name


# IndustrialDatasetHCSScenarioProvider
def test_industrial_dataset_hcs_scenario_provider(mock_csv_data, mock_variants, tmp_path):
    """Test IndustrialDatasetHCSScenarioProvider with variants, ensuring available_time is computed correctly."""
    csv_file = tmp_path / "testcases.csv"
    var_file = tmp_path / "variants.csv"
    mock_csv_data.write_csv(csv_file, separator=";")
    mock_variants.write_csv(var_file, separator=";")

    # Initialize provider with sched_time_ratio = 0.5
    provider = IndustrialDatasetHCSScenarioProvider(str(csv_file), str(var_file), sched_time_ratio=0.5)
    scenario = next(provider)  # Retrieve the first scenario

    # Debugging output
    print("Total Build Duration:", scenario.total_build_duration)
    print("Expected Available Time:", 0.5 * scenario.total_build_duration)
    print("Scenario Available Time:", scenario.get_available_time())

    # Verify total duration and available time
    assert abs(scenario.get_available_time() - 0.5 * scenario.total_build_duration) < 1e-6, (
        f"Available time should be half of total duration. "
        f"Expected: {0.5 * scenario.total_build_duration}, Got: {scenario.get_available_time()}"
    )
    assert scenario.get_variants().height > 0, "Variants should be non-empty."


# IndustrialDatasetContextScenarioProvider
def test_industrial_dataset_context_scenario_provider(mock_csv_data, tmp_path):
    """Test IndustrialDatasetContextScenarioProvider."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.write_csv(csv_file, separator=";")

    provider = IndustrialDatasetContextScenarioProvider(
        str(csv_file),
        feature_group_name="context",
        feature_group_values=["CalcPrio", "Duration"],
        previous_build=["Duration"],
    )
    scenario = next(provider)

    assert scenario.get_features() == ["CalcPrio", "Duration"], "Features should match."
    assert scenario.get_context_features().height > 0, "Context features should not be empty."


def test_context_loader_uses_previous_existing_build_when_sparse(tmp_path):
    """Context merging must use previous existing build, not (BuildId - 1)."""
    parquet_file = tmp_path / "context_sparse_builds.parquet"
    pl.DataFrame(
        {
            "BuildId": [10, 20],
            "Name": ["TC1", "TC1"],
            "Duration": [4.0, 7.0],
            "CalcPrio": [0, 0],
            "LastRun": ["2023-01-01", "2023-01-02"],
            "Verdict": [1, 1],
        }
    ).write_parquet(parquet_file)

    loader = IndustrialDatasetContextScenarioProvider(
        str(parquet_file),
        feature_group_name="context",
        feature_group_values=["Duration"],
        previous_build=["Duration"],
        sched_time_ratio=0.5,
    )

    first = next(loader)
    assert first.get_context_features()["Duration"][0] == 1

    second = next(loader)
    assert second.get_context_features()["Duration"][0] == pytest.approx(4.0)


def test_hcs_loader_variants_cache_is_bounded(tmp_path):
    """Variants cache must stay bounded to prevent unbounded memory growth."""
    tc_file = tmp_path / "testcases.parquet"
    variants_file = tmp_path / "variants.parquet"

    # Create datasets with multiple builds and variants
    pl.DataFrame(
        {
            "BuildId": [1, 2, 3, 4, 5],
            "Name": ["A", "A", "A", "A", "A"],
            "Duration": [1.0, 1.0, 1.0, 1.0, 1.0],
            "CalcPrio": [0, 0, 0, 0, 0],
            "LastRun": ["2023-01-01"] * 5,
            "Verdict": [0, 0, 0, 0, 0],
        }
    ).write_parquet(tc_file)

    pl.DataFrame(
        {
            "BuildId": [1, 2, 3, 4, 5],
            "Variant": ["V1", "V2", "V3", "V4", "V5"],
            "VarValue": [10, 20, 30, 40, 50],
        }
    ).write_parquet(variants_file)

    loader = HCSScenarioLoader(str(tc_file), str(variants_file), sched_time_ratio=0.5)
    # Access variants for multiple builds by iterating through scenarios
    for count, _ in enumerate(loader, start=1):
        if count >= 5:
            break

    # Cache size should be bounded
    assert len(loader._variants_cache) <= loader._variants_cache_size  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "relative_path",
    [
        "alibaba@druid/features-engineered.parquet",
        "square@retrofit/features-engineered.parquet",
        "fakedata/features-engineered.parquet",
        "core@dune-common/dune@debian_10 clang-7-libcpp-17/features-engineered.parquet",
        "core@dune-common/dune@debian_11 gcc-10-20/features-engineered.parquet",
        "core@dune-common/dune@ubuntu_20_04 clang-10-20/features-engineered.parquet",
        "core@dune-common/dune@total/features-engineered.parquet",
    ],
)
def test_smoke_examples_parquet_base_scenarios(relative_path):
    """Smoke test: every base examples parquet can be loaded by ScenarioLoader."""
    tcfile = Path("examples") / relative_path
    assert tcfile.exists(), f"Missing example dataset file: {tcfile}"

    provider = IndustrialDatasetScenarioProvider(str(tcfile), sched_time_ratio=0.5)
    scenario = next(provider)

    assert isinstance(scenario, VirtualScenario)
    assert len(scenario.get_testcases()) > 0
    assert scenario.total_build_duration >= 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "alibaba@druid/features-engineered.parquet",
        "square@retrofit/features-engineered.parquet",
        "fakedata/features-engineered.parquet",
        "core@dune-common/dune@debian_10 clang-7-libcpp-17/features-engineered.parquet",
        "core@dune-common/dune@debian_11 gcc-10-20/features-engineered.parquet",
        "core@dune-common/dune@ubuntu_20_04 clang-10-20/features-engineered.parquet",
        "core@dune-common/dune@total/features-engineered.parquet",
    ],
)
def test_smoke_examples_parquet_base_scenarios_duplicate(relative_path):
    """Duplicate smoke test marker (unused)."""
    pass


@pytest.mark.parametrize(
    "relative_path",
    [
        "alibaba@druid/features-engineered-contextual.parquet",
        "square@retrofit/features-engineered-contextual.parquet",
    ],
)
def test_smoke_examples_parquet_context_scenarios(relative_path):
    """Smoke test: contextual examples parquet can be loaded by ContextScenarioLoader."""
    tcfile = Path("examples") / relative_path
    assert tcfile.exists(), f"Missing example contextual dataset file: {tcfile}"

    provider = IndustrialDatasetContextScenarioProvider(
        str(tcfile),
        feature_group_name="time_execution",
        feature_group_values=["Duration", "NumErrors"],
        previous_build=["Duration", "NumErrors"],
        sched_time_ratio=0.5,
    )
    scenario = next(provider)

    assert isinstance(scenario, VirtualContextScenario)
    assert scenario.get_context_features().height > 0
    for feature in ["Duration", "NumErrors"]:
        assert feature in scenario.get_context_features().columns


def test_smoke_examples_parquet_hcs_scenario():
    """Smoke test: HCS parquet datasets can be loaded by HCSScenarioLoader.

    Validates that:
    - The loader initialises without error.
    - At least one scenario is iterable and of the correct type.
    - The variants file is non-empty at the provider level (get_total_variants).
    - At least one build across the whole dataset carries variant rows.

    Note: individual builds may legitimately have zero variant rows,
    so the per-scenario check uses the provider-level count.
    """
    tcfile = Path("examples/core@dune-common/dune@total/features-engineered.parquet")
    variants_file = Path("examples/core@dune-common/dune@total/data-variants.parquet")

    assert tcfile.exists(), f"Missing HCS features dataset file: {tcfile}"
    assert variants_file.exists(), f"Missing HCS variants dataset file: {variants_file}"

    provider = IndustrialDatasetHCSScenarioProvider(str(tcfile), str(variants_file), sched_time_ratio=0.5)

    # Provider-level variant count comes from the whole file, not per-build.
    assert provider.get_total_variants() > 0, "data-variants.parquet has no unique variants"

    scenario = next(provider)
    assert isinstance(scenario, VirtualHCSScenario)

    # At least one build across the dataset must contain variant rows.
    variants_found = scenario.get_variants().height > 0
    for sc in provider:
        if sc.get_variants().height > 0:
            variants_found = True
            break
    assert variants_found, "No build in dune@total/data-variants.parquet has variant rows"


@pytest.mark.parametrize(
    "datasets_dir,dataset,use_hcs,use_context,context_config,feature_groups,expected_type",
    [
        # --- base mode: every dataset available ---
        ("examples", "alibaba@druid", False, False, {}, {}, ScenarioLoader),
        ("examples", "square@retrofit", False, False, {}, {}, ScenarioLoader),
        ("examples", "fakedata", False, False, {}, {}, ScenarioLoader),
        ("examples", "core@dune-common/dune@total", False, False, {}, {}, ScenarioLoader),
        # --- HCS mode ---
        ("examples", "core@dune-common/dune@total", True, False, {}, {}, HCSScenarioLoader),
        # --- context mode ---
        (
            "examples",
            "alibaba@druid",
            False,
            True,
            {"previous_build": ["Duration", "NumErrors"]},
            {"feature_group_name": "time_execution", "feature_group_values": ["Duration", "NumErrors"]},
            ContextScenarioLoader,
        ),
        (
            "examples",
            "square@retrofit",
            False,
            True,
            {"previous_build": ["Duration", "NumErrors"]},
            {"feature_group_name": "time_execution", "feature_group_values": ["Duration", "NumErrors"]},
            ContextScenarioLoader,
        ),
    ],
)
def test_smoke_get_scenario_provider_path_resolution(
    datasets_dir, dataset, use_hcs, use_context, context_config, feature_groups, expected_type
):
    """Smoke test: get_scenario_provider resolves Parquet-first paths and returns a working provider.

    Validates that:
    - The resolved ``features-engineered`` path is a .parquet file (Parquet-first guarantee).
    - No DeprecationWarning is emitted (CSV fallback did NOT occur).
    - The returned provider is of the expected concrete type.
    - The provider successfully yields at least one scenario.
    """
    expected_tc_path = Path(datasets_dir) / dataset / "features-engineered.parquet"
    assert expected_tc_path.exists(), (
        f"Parquet-first assumption violated: {expected_tc_path} not found. "
        "Run the dataset conversion scripts or check examples/ layout."
    )

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = get_scenario_provider(
            datasets_dir=datasets_dir,
            dataset=dataset,
            sched_time_ratio=0.5,
            use_hcs=use_hcs,
            use_context=use_context,
            context_config=context_config,
            feature_groups=feature_groups,
        )

    # No DeprecationWarning should be emitted; that would mean CSV fallback occurred.
    csv_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning) and "CSV" in str(w.message)]
    assert not csv_warnings, (
        f"get_scenario_provider fell back to CSV despite a .parquet file being available: {csv_warnings}"
    )

    assert isinstance(provider, expected_type), f"Expected {expected_type.__name__}, got {type(provider).__name__}"

    scenario = next(iter(provider))
    assert scenario is not None


# ------------------------ Benchmark Tests ------------------------


@pytest.mark.benchmark(group="scenarios")
def test_scenario_provider_benchmark(mock_csv_data, tmp_path, benchmark):
    """Benchmark for scenario provider performance."""
    csv_file = tmp_path / "testcases.csv"
    mock_csv_data.write_csv(csv_file, separator=";")

    provider = IndustrialDatasetScenarioProvider(str(csv_file))

    def retrieve_all_scenarios():
        return list(provider)

    benchmark(retrieve_all_scenarios)


@pytest.mark.benchmark(group="scenarios")
def test_virtual_scenario_reset_benchmark(mock_testcases, benchmark):
    """Benchmark for resetting a VirtualScenario."""
    scenario = VirtualScenario(10, mock_testcases, 1, 6.5)

    benchmark(scenario.reset)


# ------------------------ Benchmark Tests ------------------------


# VirtualScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_scenario_reset(large_testcases, benchmark):
    """Benchmark resetting a large number of test cases in VirtualScenario."""
    scenario = VirtualScenario(available_time=100, testcases=large_testcases, build_id=1, total_build_duration=50000)
    benchmark(scenario.reset)


# VirtualHCSScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_hcs_scenario_get_variants(large_testcases, large_variants, benchmark):
    """Benchmark getting variants in VirtualHCSScenario."""
    scenario = VirtualHCSScenario(
        available_time=100, testcases=large_testcases, build_id=1, total_build_duration=50000, variants=large_variants
    )
    benchmark(scenario.get_variants)


# VirtualContextScenario
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_virtual_context_scenario_get_features(large_testcases, benchmark):
    """Benchmark retrieving features from a VirtualContextScenario."""
    context_features = pl.DataFrame(
        {
            "Name": [f"TC{i}" for i in range(10000)],
            "feat1": [i % 3 for i in range(10000)],
            "feat2": [i % 5 for i in range(10000)],
        }
    )
    scenario = VirtualContextScenario(
        available_time=100,
        testcases=large_testcases,
        build_id=1,
        total_build_duration=50000,
        feature_group="mock_group",
        features=["feat1", "feat2"],
        context_features=context_features,
    )
    benchmark(scenario.get_context_features)


# IndustrialDatasetScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_scenario_provider_iteration(large_csv_data, tmp_path, benchmark):
    """Benchmark iterating over scenarios with a large dataset."""
    csv_file = tmp_path / "testcases.csv"
    large_csv_data.write_csv(csv_file, separator=";")

    provider = IndustrialDatasetScenarioProvider(str(csv_file), sched_time_ratio=0.5)

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)


# IndustrialDatasetHCSScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_hcs_scenario_provider(large_csv_data, large_variants, tmp_path, benchmark):
    """Benchmark initialization and iteration of IndustrialDatasetHCSScenarioProvider with large data."""
    csv_file = tmp_path / "testcases.csv"
    var_file = tmp_path / "variants.csv"
    large_csv_data.write_csv(csv_file, separator=";")
    large_variants.write_csv(var_file, separator=";")

    provider = IndustrialDatasetHCSScenarioProvider(str(csv_file), str(var_file), sched_time_ratio=0.5)

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)


# IndustrialDatasetContextScenarioProvider
@pytest.mark.benchmark(group="scenarios")
def test_benchmark_industrial_context_scenario_provider(large_csv_data, tmp_path, benchmark):
    """Benchmark context scenario provider initialization and iteration with large data."""
    csv_file = tmp_path / "testcases.csv"
    large_csv_data.write_csv(csv_file, separator=";")

    provider = IndustrialDatasetContextScenarioProvider(
        str(csv_file),
        feature_group_name="context",
        feature_group_values=["CalcPrio", "Duration"],
        previous_build=["Duration"],
    )

    def iterate_provider():
        return list(provider)

    benchmark(iterate_provider)
