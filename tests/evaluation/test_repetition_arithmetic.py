"""Independent arithmetic and metamorphic checks for the shared panel mean."""

import math
import os
import subprocess
import sys

import pytest

from coleman.evaluation._repetition import complete_mean, repetition_execution_id


def test_equal_member_weight_is_separate_from_the_number_of_repetitions():
    assert complete_mean(((0.0,), (1.0, 1.0, 1.0))) == 0.5


def test_observed_zero_signed_values_and_missing_required_values_stay_distinct():
    assert complete_mean(((-1.0, 0.0), (0.0, 1.0))) == 0.0
    assert complete_mean(((-1.0, -1.0), (0.0, 0.0))) == -0.5
    assert complete_mean(((0.0, None), (0.0, 0.0))) is None


def test_finite_source_values_do_not_overflow_an_intermediate_float_sum():
    maximum = sys.float_info.max
    assert complete_mean(((maximum, maximum),)) == maximum
    assert complete_mean(((maximum, maximum), (-maximum, -maximum))) == 0.0


def test_final_binary64_rounding_retains_the_legacy_subnormal_envelope():
    smallest = float.fromhex("0x0.0000000000001p-1022")
    assert complete_mean(((smallest, smallest),)) == smallest
    assert complete_mean(((0.0, smallest),)) == 0.0


@pytest.mark.parametrize("panel", [(), ((),)])
def test_empty_legacy_helper_inputs_keep_their_existing_error(panel):
    with pytest.raises(ZeroDivisionError):
        complete_mean(panel)


@pytest.mark.parametrize("exponent", [-1073, -1000, -100, 0, 100, 800])
@pytest.mark.parametrize("width", [1, 3, 7])
def test_reordering_seed_duplication_and_sign_changes_preserve_the_mean_relations(exponent, width):
    panel = (
        tuple(math.ldexp(float(index + 1), exponent) for index in range(width)),
        (math.ldexp(-3.0, exponent), math.ldexp(5.0, exponent)),
    )
    expected = complete_mean(panel)
    assert expected is not None
    assert complete_mean(tuple(tuple(reversed(values)) for values in reversed(panel))) == expected
    assert complete_mean(tuple(values * 2 for values in panel)) == expected
    assert complete_mean(tuple(tuple(-value for value in values) for values in panel)) == -expected


def test_existing_ci_private_import_keeps_the_shared_implementation():
    from coleman.corpus.repeated import _complete_mean

    assert _complete_mean is complete_mean
    assert _complete_mean(builds=((0.0,), (1.0,))) == 0.5


def test_existing_ci_identity_import_keeps_its_namespace_and_signature():
    from coleman.corpus.repeated_contracts import repetition_execution_id as legacy_id
    from coleman.evaluation.design.cells import repetition_execution_id as public_id

    assert legacy_id is repetition_execution_id
    assert public_id is repetition_execution_id
    assert repetition_execution_id(candidate_id="candidate-a", seed=11) == (
        "sha256:850b0988e8cf2bd57453ba89ec4506f4dcf5f8895eeaee21b3442615edd8b7a4"
    )


def test_shared_repetition_import_does_not_load_ci_or_storage_stacks(tmp_path):
    script = """
import importlib.abc,sys
blocked=('coleman.artifacts','coleman.corpus','coleman.cli','coleman.mcp',
         'coleman.application','coleman.runner','sqlite3','numpy','scipy','pyarrow','polars')
preloaded=sorted(name for name in sys.modules if any(name==item or name.startswith(item+'.') for item in blocked))
assert not preloaded, 'Cold interpreter unexpectedly preloaded blocked dependencies: '+repr(preloaded)
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
   raise AssertionError('Pure repetition crossed import boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.evaluation._repetition import complete_mean,repetition_execution_id
assert complete_mean(((0.0,),(1.0,)))==0.5
assert repetition_execution_id('candidate-a',11).endswith('edd8b7a4')
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        # Coverage's startup hook imports SQLite before this script can install
        # its finder. Keep this actual cold-import check free of instrumentation;
        # the arithmetic above is covered normally in the pytest process.
        env={key: value for key, value in os.environ.items() if not key.startswith(("COVERAGE_", "COV_CORE_"))},
    )
    assert result.returncode == 0, result.stdout + result.stderr
