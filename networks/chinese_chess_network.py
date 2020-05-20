import math
import copy
import numpy as np
import torch
from torch import nn as nn

from game.game import Action
from networks.network import BaseNetwork
from networks.resnet import RepresentationNetwork,DynamicNetwork,PredictionNetwork


class ChineseChessNetwork(BaseNetwork):

    def __init__(self,
                 observatin_size: int,
                 action_size: int,
                 num_channel: int,
        ):
        self.observatin_size = observatin_size
        self.action_size = action_size

        """
        regularizer = regularizers.l2(weight_decay)
        representation_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(representation_size, activation=representation_activation,
                                                   kernel_regularizer=regularizer)])
        value_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])
        policy_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(representation_size, activation=representation_activation,
                                            kernel_regularizer=regularizer)])
        """
        middle_size = copy.copy(observatin_size)
        middle_size[0] = num_channel
        representation_network = RepresentationNetwork(
            observation_shape=observatin_size,
            num_blocks=4,
            num_channels=num_channel,
        )
        representation_network = representation_network.cuda()

        value_policy_network = PredictionNetwork(
            observation_shape=middle_size,
            action_space_size=action_size,
            num_blocks=2,
            num_channels=num_channel,
            reduced_channels=4,
            fc_value_layers=[64,],
            fc_policy_layers=[64,],
        )
        value_policy_network = value_policy_network.cuda()

        dynamic_size = copy.copy(observatin_size)
        dynamic_size[0] = num_channel + 2
        dynamic_network = DynamicNetwork(
            observation_shape=dynamic_size,
            num_blocks=2,
            num_channels=num_channel,
        )
        dynamic_network = dynamic_network.cuda()

        super().__init__(representation_network, value_policy_network, dynamic_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = float(np.tanh(value_support[0][0]))
        return value

    def _conditioned_hidden_state(self, hidden_state: torch.Tensor, action: int) -> np.array:
        #hidden_state = hidden_state.cpu().detach().numpy()
        action_plane = np.zeros((2,10,9))
        action_ind = action
        from_ind,to_ind = divmod(action_ind,90)
        def set_plane_onehot(action_plane,pos,plane_id):
            y,x = divmod(pos,9)
            y = 9 - y
            action_plane[plane_id,y,x] = 1
        set_plane_onehot(action_plane,from_ind,0)
        set_plane_onehot(action_plane,to_ind,1)

        conditioned_hidden = np.concatenate((hidden_state, action_plane),axis=0)
        return np.expand_dims(conditioned_hidden, axis=0)
