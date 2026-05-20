"""Agent classes for the Coleman framework.

This module provides an abstract representation of an agent in the Coleman framework.

An `Agent` represents an entity that interacts with the environment to perform
test case prioritization. The agent uses a policy to decide on an action (i.e.,
a prioritized list of test cases) and then observes the environment to receive a reward.
The agent updates its internal state or knowledge based on the reward, allowing it to
improve its decisions over time.

Classes
-------
Agent
    Base class for agents. Defines common methods and properties all agents should have.
RewardAgent
    An agent that learns using a reward function. Inherits from `Agent`.
ContextualAgent
    Extends the `RewardAgent` to incorporate contextual information for decision-making.
RewardSlidingWindowAgent
    An agent that learns using a sliding window mechanism and a reward function.
    Inherits from `RewardAgent`.
SlidingWindowContextualAgent
    Combines the sliding window mechanism with contextual information.
    Inherits from `RewardAgent`.

Notes
-----
Common attributes across agent types:

- ``policy``: The policy used by the agent to choose an action.
- ``bandit``: An instance of the Bandit class that the agent interacts with.
- ``actions``: A DataFrame that tracks the agent's actions and their respective outcomes.
- ``last_prioritization``: Stores the last action chosen by the agent.
- ``t``: Represents the time or the number of steps the agent has taken.
- ``context_features``: (For contextual agents) Contains the features of the context.
- ``history``: (For sliding window agents) Maintains a history of actions taken by the agent.
- ``window_size``: (For sliding window agents) Determines the size of the sliding window.
"""

from .base import Agent
from .contextual import ContextualAgent
from .reward_agent import RewardAgent
from .schemas import ACTIONS_SCHEMA, HISTORY_SCHEMA
from .sliding_window import RewardSlidingWindowAgent, SlidingWindowContextualAgent

__all__ = [
    "ACTIONS_SCHEMA",
    "HISTORY_SCHEMA",
    "Agent",
    "RewardAgent",
    "ContextualAgent",
    "RewardSlidingWindowAgent",
    "SlidingWindowContextualAgent",
]
