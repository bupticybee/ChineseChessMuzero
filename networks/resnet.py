import torch
from torch import nn as nn
import numpy as np
########### End ResNet ###########
##################################


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super().__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(size_list[i], size_list[i + 1]),
                        torch.nn.LeakyReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        for block in self.resblocks1:
            out = block(out)
        out = self.conv2(out)
        for block in self.resblocks2:
            out = block(out)
        out = self.pooling1(out)
        for block in self.resblocks3:
            out = block(out)
        out = self.pooling2(out)
        return out

def format_input(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float().cuda()
    else:
        return x

class RepresentationNetwork(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            num_blocks,
            num_channels,
    ):
        super().__init__()
        self.conv = conv3x3(
            observation_shape[0] ,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = format_input(x)
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        return out.cpu().detach().numpy()


class DynamicNetwork(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            num_blocks,
            num_channels,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.conv = conv3x3(self.observation_shape[0], num_channels)
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = format_input(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        return state.cpu().detach().numpy()


class PredictionNetwork(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels,
            fc_value_layers,
            fc_policy_layers,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1 = torch.nn.Conv2d(num_channels, reduced_channels, 1)
        self.block_output_size = reduced_channels
        for one_shape in observation_shape[1:]:
            self.block_output_size *= one_shape
        self.fc_value = FullyConnectedNetwork(
            self.block_output_size, fc_value_layers, 1, activation=None,
        )
        self.fc_policy = FullyConnectedNetwork(
            self.block_output_size,
            fc_policy_layers,
            action_space_size,
            activation=None,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = format_input(x)
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.conv1x1(out)
        out = out.view(-1, self.block_output_size)
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        policy = self.softmax(policy)
        return policy.cpu().detach().numpy(), value.cpu().detach().numpy()

