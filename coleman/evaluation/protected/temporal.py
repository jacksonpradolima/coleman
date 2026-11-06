"""Compatibility aliases for unchanged point-in-time split contracts.

The implementation lives in the independently importable core. Existing class
identities, validation, fields and serialized output retain their v1 meaning.
"""

from coleman.core.splits import (
    Partition as Partition,
)
from coleman.core.splits import (
    SplitDecision as SplitDecision,
)
from coleman.core.splits import (
    TemporalObservation as TemporalObservation,
)
from coleman.core.splits import (
    TemporalPlan as TemporalPlan,
)
from coleman.core.temporal import Timestamp as Timestamp
