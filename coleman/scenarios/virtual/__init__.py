"""Virtual scenario classes."""

from .scenarios import VirtualContextScenario, VirtualHCSScenario, VirtualScenario

# Short-name aliases
Scenario = VirtualScenario
HCSScenario = VirtualHCSScenario
ContextScenario = VirtualContextScenario

__all__ = [
    "VirtualScenario",
    "VirtualHCSScenario",
    "VirtualContextScenario",
    "Scenario",
    "HCSScenario",
    "ContextScenario",
]
