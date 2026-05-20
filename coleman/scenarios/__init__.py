"""
coleman.scenarios - Scenario Management for the Coleman Framework.

This module provides utilities for managing and processing different scenarios in the context
of the Coleman framework. This includes virtual scenarios for commits, scenarios specific
to HCS context, and scenarios that consider context information for each commit. The module
also provides utilities to load dataset files for experimental evaluations.

Classes
-------
VirtualScenario
    Basic virtual scenario to manipulate data for each commit.
VirtualHCSScenario
    Extends VirtualScenario to handle HCS context.
VirtualContextScenario
    Extends VirtualScenario to handle context information.
ScenarioLoader
    Loader that reads a build dataset and yields scenarios.
HCSScenarioLoader
    Extends ScenarioLoader to handle HCS scenarios.
ContextScenarioLoader
    Extends ScenarioLoader to handle context scenarios.

Short Aliases
-------------
Scenario
    Alias for VirtualScenario.
HCSScenario
    Alias for VirtualHCSScenario.
ContextScenario
    Alias for VirtualContextScenario.

Backward-Compatible Names
-------------------------
IndustrialDatasetScenarioProvider
    Alias for ScenarioLoader.
IndustrialDatasetHCSScenarioProvider
    Alias for HCSScenarioLoader.
IndustrialDatasetContextScenarioProvider
    Alias for ContextScenarioLoader.
"""

from .loaders import (
    ContextScenarioLoader,
    HCSScenarioLoader,
    IndustrialDatasetContextScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
    ScenarioLoader,
)
from .virtual import (
    ContextScenario,
    HCSScenario,
    Scenario,
    VirtualContextScenario,
    VirtualHCSScenario,
    VirtualScenario,
)

# Keep old ScenarioProvider short aliases for backward compatibility
ScenarioProvider = ScenarioLoader
HCSScenarioProvider = HCSScenarioLoader
ContextScenarioProvider = ContextScenarioLoader

__all__ = [
    # Virtual scenario classes
    "VirtualScenario",
    "VirtualHCSScenario",
    "VirtualContextScenario",
    # Short aliases for virtual scenarios
    "Scenario",
    "HCSScenario",
    "ContextScenario",
    # Primary loader names
    "ScenarioLoader",
    "HCSScenarioLoader",
    "ContextScenarioLoader",
    # Backward-compatible names
    "IndustrialDatasetScenarioProvider",
    "IndustrialDatasetHCSScenarioProvider",
    "IndustrialDatasetContextScenarioProvider",
    "ScenarioProvider",
    "HCSScenarioProvider",
    "ContextScenarioProvider",
]
