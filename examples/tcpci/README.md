# A complete CI decision, without a native runner

Copy `ci.py` and `model.py` beside each other outside the checkout, install the
reviewed COLEMAN wheel, then run `python -I -B ci.py` or
`python -I -B ci.py --full`.

The example uses the actual Hedge implementation, keeps its original full
expert feedback separate from selected test verdicts, maps an explicit ranking
to an exact modeled-budget prefix, and independently solves finite coverage.
All source outcomes are synthetic; physical elapsed time stays unknown. No
adaptive win, probability calibration or durable ACK is claimed.

The [human guide](../../docs/tcpci-problems.md) explains the input table, complete
cycle, output, sampling limits and the difference between a prefix and a set.
