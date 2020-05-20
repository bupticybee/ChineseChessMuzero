import typing
from abc import ABC, abstractmethod
from typing import Dict, List, Callable

import numpy as np
from torch.nn import Module
# TODO 替换这里的tensorflow模块并继续完成ChineseChessNetwork

from game.game import Action


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: typing.Optional[List[float]]

    @staticmethod
    def build_policy_logits(policy_logits):
        return policy_logits[0].tolist()#{Action(i): logit for i, logit in enumerate(policy_logits[0])}


class AbstractNetwork(ABC):

    def __init__(self):
        self.training_steps = 0

    @abstractmethod
    def initial_inference(self, image) -> NetworkOutput:
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        pass


class UniformNetwork(AbstractNetwork):
    """policy -> uniform, value -> 0"""

    def __init__(self, action_size: int):
        super().__init__()
        self.action_size = action_size

    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0,  {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        return NetworkOutput(0, 0,  {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)


class InitialModule(Module):
    """Module that combine the representation and prediction (value+policy) network."""

    def __init__(self, representation_network: Module, value_policy_network: Module):
        super(InitialModule, self).__init__()
        self.representation_network = representation_network
        self.value_policy_network = value_policy_network

    def forward(self, image):
        hidden_representation = self.representation_network(image)
        policy_logits,value = self.value_policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class RecurrentModule(Module):
    """Module that combine the dynamic and prediction (value+policy) network."""

    def __init__(self, dynamic_network: Module, value_policy_network: Module):
        super(RecurrentModule, self).__init__()
        self.dynamic_network = dynamic_network
        self.value_policy_network = value_policy_network

    def forward(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        policy_logits,value = self.value_policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class BaseNetwork(AbstractNetwork):
    """Base class that contains all the networks and models of MuZero."""

    def __init__(self, representation_network: Module, value_policy_network: Module,
                 dynamic_network: Module):
        super().__init__()
        # Networks blocks
        self.representation_network = representation_network
        self.value_policy_network = value_policy_network
        self.dynamic_network = dynamic_network

        # Modules for inference and training
        self.initial_model = InitialModule(self.representation_network, self.value_policy_network)
        self.recurrent_model = RecurrentModule(self.dynamic_network, self.value_policy_network)

    def initial_inference(self, image: np.array) -> NetworkOutput:
        """representation + prediction function"""

        hidden_representation, value, policy_logits = self.initial_model(np.expand_dims(image, 0))
        output = NetworkOutput(value=self._value_transform(value),
                               reward=0,
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    def recurrent_inference(self, hidden_state: np.array, action: int) -> NetworkOutput:
        """dynamics + prediction function"""

        conditioned_hidden = self._conditioned_hidden_state(hidden_state, action)
        hidden_representation, value, policy_logits = self.recurrent_model(conditioned_hidden)
        output = NetworkOutput(value=self._value_transform(value),
                               reward=0,
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    @abstractmethod
    def _value_transform(self, value: np.array) -> float:
        pass

    @abstractmethod
    def _conditioned_hidden_state(self, hidden_state: np.array, action: int) -> np.array:
        pass

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables
