"""Keep the executable notebook consistent with native fault-detection units."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from coleman.evaluation.apfdc import APFDcMetric


def test_actual_stability_cell_uses_higher_apfdc_to_break_a_napfd_tie():
    suite = [
        {"Name": "fails", "Duration": 1.0, "NumErrors": 1, "Verdict": 1},
        {"Name": "passes", "Duration": 1.0, "NumErrors": 0, "Verdict": 0},
    ]
    first, last = APFDcMetric(), APFDcMetric()
    for metric, ordering in ((first, suite), (last, list(reversed(suite)))):
        metric.update_available_time(2.0)
        metric.evaluate(ordering)
    # Independently: the fault receives 1.5/2 versus 0.5/2 cost-aware credit.
    assert first.cost == 0.75
    assert last.cost == 0.25
    summary = pd.DataFrame(
        {
            "policy": ["later-fault", "earlier-fault", "lower-napfd"],
            "reward_function": ["RNFail"] * 3,
            "execution_id": ["one", "two", "three"],
            "avg_napfd": [0.8, 0.8, 0.7],
            "avg_apfdc": [last.cost, first.cost, 1.0],
        }
    )
    source = Path(__file__).parents[1] / "docs/workflow.py"
    cells = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and [argument.arg for argument in node.args.args] == ["mo", "pd", "summary_df"]
    ]
    assert len(cells) == 1
    cell = cells[0]
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(source), "exec"), namespace)
    # Only Markdown rendering is replaced; the notebook's actual grouping/sort
    # runs on pandas with actual native scores, without importing the UI app.
    actual = namespace["_"](SimpleNamespace(md=str), pd, summary)
    assert actual["policy"].tolist() == ["earlier-fault", "later-fault", "lower-napfd"]
