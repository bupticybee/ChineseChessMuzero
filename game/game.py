from abc import abstractmethod, ABC
from typing import List

from self_play.utils import Node
import numpy as np
from collections import namedtuple


class Action(namedtuple("Action","index")):
    """ Class that represent an action of a game."""

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class Player(object):
    """
    A one player class.
    This class is useless, it's here for legacy purpose and for potential adaptations for a two players MuZero.
    """
    def __init__(self,player=0):
        self.player = player

    def __eq__(self, other):
        return True


class ActionHistory(object):
    """
    Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        # TODO 这里乱copy还得了，太慢了,这个必须优化
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()


class AbstractGame(ABC):
    """
    Abstract class that allows to implement a game.
    One instance represent a single episode of interaction with the environment.
    """

    def __init__(self, discount: float):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def apply(self, action: Action):
        """Apply an action onto the environment."""

        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        """After each MCTS run, store the statistics generated by the search."""

        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        """Generate targets to learn from during the network training."""

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        """Return the current player."""
        return Player()

    def action_history(self) -> ActionHistory:
        """Return the actions executed inside the search."""
        return ActionHistory(self.history, self.action_space_size)

    # Methods to be implemented by the children class
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        pass

    @abstractmethod
    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """Is the game is finished?"""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        pass

    @abstractmethod
    def make_observation(self, state_index: int):
        """Compute the state of the game."""
        pass

    @abstractmethod
    def make_observation_image(self, state_index: int):
        """Compute the state of the game."""
        pass
