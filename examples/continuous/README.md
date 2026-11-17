# A forecast is an input to a decision

This public synthetic example uses the **unreleased v5 source wheel**. It learns
signed responses from eligible labels, then uses an independent constrained
solver. It keeps forecasting error, solver objective, coefficient movement and
own-book modeled charges separate. No principal component is mocked.

| File | Use |
| --- | --- |
| `model.py` | Real append-only ridge host, original pending features and pre-label controls |
| `flow.py` | Existing retained causal replay, chronological labels and complete32-child state bindings |
| `allocation.py` | Coupled convex optimization, projected online steps and fixed/periodic/finite controls |

From the repository, build and install a noneditable wheel, copy the public
example and execute it outside the checkout:

```bash
uv build --no-build-isolation --out-dir /tmp/coleman-continuous-dist
uv venv /tmp/coleman-continuous-env --python 3.14
uv pip install --python /tmp/coleman-continuous-env/bin/python /tmp/coleman-continuous-dist/coleman-4.20.0-py3-none-any.whl
mkdir -p /tmp/coleman-continuous-demo
cp examples/continuous/model.py examples/continuous/flow.py examples/continuous/allocation.py /tmp/coleman-continuous-demo/
cd /tmp/coleman-continuous-demo
/tmp/coleman-continuous-env/bin/python -I -B flow.py
/tmp/coleman-continuous-env/bin/python -I -B allocation.py
```

The integration wheel retains4.20.0 metadata until the single final v5 release;
it is distinct from the published4.20.0 package. Record the reviewed wheel hash
and source revision in any experiment. There are no optional analytics or CI
dependencies in the minimal forecasting/optimization route.

The first three ideal predictions are0,1,6/5. The zero second label updates the
Gram matrix. Ridge loses to the fixed zero forecast on this authored path; all
results remain descriptive with insufficient evidence. Coupled optimization
independently returns `(0.5,1.25)` at objective`-2.8125` for the worked active-box
fixture, while coordinate clipping gives the inferior feasible `(0.5,1)`.

See the [human guide](../../docs/continuous-decisions.md) for complete input/output
tables, assumptions, numerical envelopes, recovery semantics and primary sources.
Signed mathematical coefficients do not establish financial shorting, funding,
collateral, margin support or market predictability.
