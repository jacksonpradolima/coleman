"""
Unit tests for the coleman evaluation module.

This test suite validates the correctness and robustness of the evaluation metrics
implemented in the coleman library, including NAPFDMetric and NAPFDVerdictMetric.

Tests cover various scenarios, including:
- Standard test cases with faults and verdicts.
- Handling of edge cases like empty records and no failures.
- Common behaviors such as string representation and abstract class enforcement.
- Metrics computation with records having identical durations or results.

Fixtures:
- `sample_records`: Provides a set of sample test cases with varying durations, errors, and verdicts.
- `available_time`: Computes the total available time from the sample records.

Constants:
- `NAPFD_FITNESS_NON_NEGATIVE`: Ensures that the NAPFD fitness value is non-negative.
- `NAPFD_FITNESS_NOT_EXCEED_ONE`: Ensures that the NAPFD fitness value does not exceed 1.
- `NAPFD_COST_NON_NEGATIVE`: Ensures that the NAPFD cost value is non-negative.

Helper Functions:
- `_common_test_napfd`: A shared utility to test NAPFDMetric across multiple scenarios.

Coverage:
- Ensures all methods, including edge cases, are well-tested to maintain high reliability.

Usage:
Run the tests using pytest to verify the functionality of evaluation metrics.
"""

import math

import polars as pl
import pytest

from coleman.evaluation import (
    APFDcMetric,
    AveragePrecisionAtKMetric,
    EvaluationMetric,
    NAPFDMetric,
    NAPFDVerdictMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    ReciprocalRankAtKMetric,
)

# Constants for error messages
NAPFD_FITNESS_NON_NEGATIVE = "NAPFD fitness should be non-negative."
NAPFD_FITNESS_NOT_EXCEED_ONE = "NAPFD fitness should not exceed 1."
NAPFD_COST_NON_NEGATIVE = "NAPFD cost should be non-negative."


@pytest.fixture
def sample_records():
    """
    Provide sample test records for evaluation metrics.
    """
    return [
        {"Name": 8, "Duration": 0.001, "NumRan": 1, "NumErrors": 3, "Verdict": 1},
        {"Name": 9, "Duration": 0.497, "NumRan": 1, "NumErrors": 1, "Verdict": 1},
        {"Name": 4, "Duration": 0.188, "NumRan": 3, "NumErrors": 2, "Verdict": 1},
        {"Name": 6, "Duration": 0.006, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": 3, "Duration": 0.006, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": 1, "Duration": 0.235, "NumRan": 2, "NumErrors": 0, "Verdict": 0},
        {"Name": 2, "Duration": 5.704, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": 5, "Duration": 3.172, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": 7, "Duration": 0.034, "NumRan": 1, "NumErrors": 5, "Verdict": 1},
    ]


@pytest.fixture
def available_time(sample_records):
    """
    Calculate total available time from sample records.
    """
    return sum(item["Duration"] for item in sample_records)


def test_evaluation_metric_not_implemented():
    """
    Test that the abstract EvaluationMetric class raises NotImplementedError.
    """
    metric = EvaluationMetric()
    with pytest.raises(NotImplementedError):
        metric.evaluate([])


def test_evaluation_metric_str():
    """
    Test that the __str__ method is implemented for derived metric classes.
    """
    napfd = NAPFDMetric()
    napfd_v = NAPFDVerdictMetric()
    apfdc = APFDcMetric()

    assert str(napfd) == "NAPFD", "NAPFDMetric __str__ method failed."
    assert str(napfd_v) == "NAPFDVerdict", "NAPFDVerdictMetric __str__ method failed."
    assert str(apfdc) == "APFDc", "APFDcMetric __str__ method failed."


def test_topk_metrics_string_names():
    """Top-k metric classes should expose stable string names."""
    assert str(PrecisionAtKMetric(top_k=6)) == "PrecisionAtK"
    assert str(RecallAtKMetric(top_k=6)) == "RecallAtK"
    assert str(AveragePrecisionAtKMetric(top_k=6)) == "AveragePrecisionAtK"
    assert str(ReciprocalRankAtKMetric(top_k=6)) == "ReciprocalRankAtK"
    assert str(NDCGAtKMetric(top_k=6)) == "NDCGAtK"


def test_topk_precision_recall_ap_rr_metrics():
    """Validate common IR top-k metrics over failure verdicts."""
    records = [
        {"Name": "T1", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T2", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T3", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T4", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T5", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T6", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T7", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
    ]

    precision_k = PrecisionAtKMetric(top_k=6)
    recall_k = RecallAtKMetric(top_k=6)
    ap_k = AveragePrecisionAtKMetric(top_k=6)
    rr_k = ReciprocalRankAtKMetric(top_k=6)

    for metric in (precision_k, recall_k, ap_k, rr_k):
        metric.update_available_time(10_000)
        metric.evaluate(records)

    # top-6 contains 4 failures => P@6 = 4/6
    assert precision_k.fitness == pytest.approx(4 / 6)
    # total failures are 4 and all of them are in top-6 => R@6 = 1
    assert recall_k.fitness == pytest.approx(1.0)
    # AP@6 with failing ranks [1,3,5,6]:
    # (1/1 + 2/3 + 3/5 + 4/6) / 4
    assert ap_k.fitness == pytest.approx((1 + (2 / 3) + (3 / 5) + (4 / 6)) / 4)
    # first failure at rank 1 => RR@6 = 1
    assert rr_k.fitness == pytest.approx(1.0)


def test_topk_precision_exactly_one_for_6_out_of_6():
    """If the first 6 selected tests fail, Precision@6 should be 1.0."""
    records = [
        {"Name": f"T{i}", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1 if i <= 6 else 0}
        for i in range(1, 11)
    ]
    metric = PrecisionAtKMetric(top_k=6)
    metric.update_available_time(10_000)
    metric.evaluate(records)
    assert metric.fitness == pytest.approx(1.0)


def test_topk_ndcg_metric():
    """nDCG@k should match binary-relevance discounted gain normalization."""
    records = [
        {"Name": "T1", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T2", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T3", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T4", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T5", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T6", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T7", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
    ]

    metric = NDCGAtKMetric(top_k=6)
    metric.update_available_time(10_000)
    metric.evaluate(records)

    dcg = sum(1.0 / math.log2(rank + 1.0) for rank in [1, 3, 5, 6])
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in [1, 2, 3, 4])
    assert metric.fitness == pytest.approx(dcg / idcg)


def test_topk_ndcg_perfect_ranking_is_one():
    """nDCG@k should be 1.0 for an ideal ranking in top-k."""
    records = [
        {"Name": f"T{i}", "Duration": 1.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1 if i <= 4 else 0}
        for i in range(1, 11)
    ]
    metric = NDCGAtKMetric(top_k=6)
    metric.update_available_time(10_000)
    metric.evaluate(records)
    assert metric.fitness == pytest.approx(1.0)


def test_topk_precision_can_use_time_budget():
    """Precision@k can optionally cap the prefix by available_time."""
    records = [
        {"Name": "T1", "Duration": 2.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T2", "Duration": 2.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T3", "Duration": 2.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T4", "Duration": 2.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
    ]

    metric = PrecisionAtKMetric(top_k=4, use_time_budget=True)
    metric.update_available_time(5.0)
    metric.evaluate(records)

    assert metric.scheduled_testcases == ["T1", "T2"]
    assert metric.unscheduled_testcases == ["T3", "T4"]
    assert metric.fitness == pytest.approx(1.0)
    assert metric.recall == pytest.approx(2 / 3)


def test_topk_precision_keeps_legacy_behavior_without_budget_mode():
    """Default top-k behavior must ignore available_time for backward compatibility."""
    records = [
        {"Name": "T1", "Duration": 5.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T2", "Duration": 5.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T3", "Duration": 5.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
    ]

    metric = PrecisionAtKMetric(top_k=2)
    metric.update_available_time(1.0)
    metric.evaluate(records)

    assert metric.scheduled_testcases == ["T1", "T2"]
    assert metric.fitness == pytest.approx(0.5)


def test_evaluation_metric_set_default_metrics():
    """Cover the default-metric initializer used by derived metrics."""
    metric = EvaluationMetric()
    metric.set_default_metrics()

    assert metric.ttf == -1
    assert metric.recall == 1
    assert metric.avg_precision == 1
    assert metric.fitness == 1
    assert metric.cost == 1


def test_napfd_verdict_metric_subset_size_budget_schedules_by_k():
    """subset_size budget should schedule exactly k tests regardless of durations."""
    records = [
        {"Name": "T1", "Duration": 10.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
        {"Name": "T2", "Duration": 10.0, "NumRan": 1, "NumErrors": 0, "Verdict": 0},
        {"Name": "T3", "Duration": 10.0, "NumRan": 1, "NumErrors": 0, "Verdict": 1},
    ]

    metric = NAPFDVerdictMetric()
    metric.update_budget("subset_size", 2)
    metric.update_available_time(0.0)
    metric.evaluate(records)

    assert metric.scheduled_testcases == ["T1", "T2"]
    assert metric.unscheduled_testcases == ["T3"]


def test_napfd_metric(sample_records, available_time):
    """
    Test NAPFDMetric with standard records and 50% available time.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate(sample_records)

    assert napfd.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd.cost >= 0, NAPFD_COST_NON_NEGATIVE
    assert 0 <= napfd.avg_precision <= 1


def test_napfd_verdict_metric(sample_records, available_time):
    """
    Test NAPFDVerdictMetric with standard records and 50% available time.
    """
    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)
    napfd_v.evaluate(sample_records)

    assert napfd_v.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd_v.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd_v.cost >= 0, NAPFD_COST_NON_NEGATIVE
    assert 0 <= napfd_v.avg_precision <= 1


def test_empty_records(available_time):
    """
    Test metrics with empty records to ensure proper handling.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate([])

    assert napfd.fitness == 1, "NAPFD fitness should be 1 for empty records."
    assert napfd.cost == 1, "NAPFD cost should be 1 for empty records."


def test_napfd_verdict_metric_no_failures(available_time):
    """
    Test NAPFDVerdictMetric with no failures to ensure default metrics are set.
    """
    no_failure_records = [{"Name": i, "Duration": 1, "NumRan": 1, "NumErrors": 0, "Verdict": 0} for i in range(1, 10)]

    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)
    napfd_v.evaluate(no_failure_records)

    assert napfd_v.fitness == 1, "NAPFD-V fitness should be 1 when no failures are present."
    assert napfd_v.cost == 1, "NAPFD-V cost should be 1 when no failures are present."


def test_apfdc_metric(sample_records, available_time):
    """APFDcMetric should expose the cost-aware score explicitly."""
    metric = APFDcMetric()
    metric.update_available_time(available_time * 0.5)
    metric.evaluate(sample_records)

    assert metric.cost >= 0
    assert metric.cost <= 1
    assert metric.fitness == pytest.approx(metric.cost)
    assert metric.testcase_costs == [item["Duration"] for item in sample_records]


def test_evaluation_metric_as_suite_frame_empty_list():
    """Cover _as_suite_frame when test_suite is an empty list."""
    metric = NAPFDMetric()
    result = metric._as_suite_frame([], error_key="NumErrors")
    assert result.is_empty()
    assert set(result.columns) == {"Name", "Duration", "Verdict", "NumErrors"}


def test_napfd_metric_accepts_polars_dataframe(sample_records, available_time):
    """NAPFDMetric should accept a Polars DataFrame without list-of-dicts conversion."""
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate(pl.DataFrame(sample_records))

    assert napfd.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_napfd_verdict_metric_accepts_polars_dataframe(sample_records, available_time):
    """NAPFDVerdictMetric should accept a Polars DataFrame without list-of-dicts conversion."""
    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)
    napfd_v.evaluate(pl.DataFrame(sample_records))

    assert napfd_v.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd_v.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd_v.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_napfd_metric_list_and_dataframe_are_equivalent(sample_records, available_time):
    """NAPFDMetric should produce identical results for list-of-dicts and DataFrame inputs."""
    metric_list = NAPFDMetric()
    metric_df = NAPFDMetric()

    budget = available_time * 0.5
    metric_list.update_available_time(budget)
    metric_df.update_available_time(budget)

    metric_list.evaluate(sample_records)
    metric_df.evaluate(pl.DataFrame(sample_records))

    assert metric_list.scheduled_testcases == metric_df.scheduled_testcases
    assert metric_list.unscheduled_testcases == metric_df.unscheduled_testcases
    assert metric_list.detection_ranks == metric_df.detection_ranks
    assert metric_list.detection_ranks_failures == metric_df.detection_ranks_failures
    assert metric_list.detection_ranks_time == metric_df.detection_ranks_time
    assert metric_list.detected_failures == metric_df.detected_failures
    assert metric_list.undetected_failures == metric_df.undetected_failures
    assert metric_list.ttf_duration == pytest.approx(metric_df.ttf_duration)
    assert metric_list.fitness == pytest.approx(metric_df.fitness)
    assert metric_list.cost == pytest.approx(metric_df.cost)


def test_napfd_verdict_list_and_dataframe_are_equivalent(sample_records, available_time):
    """NAPFDVerdictMetric should produce identical results for list-of-dicts and DataFrame inputs."""
    metric_list = NAPFDVerdictMetric()
    metric_df = NAPFDVerdictMetric()

    budget = available_time * 0.5
    metric_list.update_available_time(budget)
    metric_df.update_available_time(budget)

    metric_list.evaluate(sample_records)
    metric_df.evaluate(pl.DataFrame(sample_records))

    assert metric_list.scheduled_testcases == metric_df.scheduled_testcases
    assert metric_list.unscheduled_testcases == metric_df.unscheduled_testcases
    assert metric_list.detection_ranks == metric_df.detection_ranks
    assert metric_list.detection_ranks_failures == metric_df.detection_ranks_failures
    assert metric_list.detection_ranks_time == metric_df.detection_ranks_time
    assert metric_list.detected_failures == metric_df.detected_failures
    assert metric_list.undetected_failures == metric_df.undetected_failures
    assert metric_list.ttf_duration == pytest.approx(metric_df.ttf_duration)
    assert metric_list.fitness == pytest.approx(metric_df.fitness)
    assert metric_list.cost == pytest.approx(metric_df.cost)


def test_napfd_metric_dataframe_matches_list_results(sample_records, available_time):
    """Vectorized DataFrame path must preserve list-input metric semantics."""
    metric_list = NAPFDMetric()
    metric_df = NAPFDMetric()
    budget = available_time * 0.5
    metric_list.update_available_time(budget)
    metric_df.update_available_time(budget)

    metric_list.evaluate(sample_records)
    metric_df.evaluate(pl.DataFrame(sample_records))

    assert metric_df.fitness == pytest.approx(metric_list.fitness)
    assert metric_df.cost == pytest.approx(metric_list.cost)
    assert metric_df.detected_failures == metric_list.detected_failures
    assert metric_df.undetected_failures == metric_list.undetected_failures
    assert metric_df.detection_ranks == metric_list.detection_ranks


def test_napfd_verdict_metric_dataframe_matches_list_results(sample_records, available_time):
    """Verdict metric should match between list and DataFrame inputs."""
    metric_list = NAPFDVerdictMetric()
    metric_df = NAPFDVerdictMetric()
    budget = available_time * 0.5
    metric_list.update_available_time(budget)
    metric_df.update_available_time(budget)

    metric_list.evaluate(sample_records)
    metric_df.evaluate(pl.DataFrame(sample_records))

    assert metric_df.fitness == pytest.approx(metric_list.fitness)
    assert metric_df.cost == pytest.approx(metric_list.cost)
    assert metric_df.detected_failures == metric_list.detected_failures
    assert metric_df.undetected_failures == metric_list.undetected_failures
    assert metric_df.detection_ranks == metric_list.detection_ranks


def _common_test_napfd(records, available_time):
    """
    Common helper function to test NAPFDMetric.
    """
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)
    napfd.evaluate(records)

    assert napfd.fitness >= 0, NAPFD_FITNESS_NON_NEGATIVE
    assert napfd.fitness <= 1, NAPFD_FITNESS_NOT_EXCEED_ONE
    assert napfd.cost >= 0, NAPFD_COST_NON_NEGATIVE


def test_identical_durations(available_time):
    """
    Test metrics with records having identical durations.
    """
    identical_records = [
        {"Name": i, "Duration": 1, "NumRan": 1, "NumErrors": i % 2, "Verdict": i % 2} for i in range(1, 10)
    ]
    _common_test_napfd(identical_records, available_time)


def test_identical_cost_and_results(available_time):
    """
    Test metrics where all records have the same cost and results.
    """
    identical_cost_records = [
        {"Name": i, "Duration": 1, "NumRan": 1, "NumErrors": 0, "Verdict": 1} for i in range(1, 10)
    ]
    _common_test_napfd(identical_cost_records, available_time)


@pytest.mark.benchmark(group="evaluation")
def test_benchmark_napfd_metric(benchmark, sample_records, available_time):
    """
    Benchmark the performance of NAPFDMetric with a large dataset.
    """
    large_dataset = sample_records * 10_000  # Simulate a large dataset
    napfd = NAPFDMetric()
    napfd.update_available_time(available_time * 0.5)

    # Benchmark the evaluation process
    benchmark(napfd.evaluate, large_dataset)


@pytest.mark.benchmark(group="evaluation")
def test_benchmark_napfd_verdict_metric(benchmark, sample_records, available_time):
    """
    Benchmark the performance of NAPFDVerdictMetric with a large dataset.
    """
    large_dataset = sample_records * 10_000  # Simulate a large dataset
    napfd_v = NAPFDVerdictMetric()
    napfd_v.update_available_time(available_time * 0.5)

    # Benchmark the evaluation process
    benchmark(napfd_v.evaluate, large_dataset)
