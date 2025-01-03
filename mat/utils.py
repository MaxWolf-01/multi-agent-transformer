from torch import nn


def init_weights(m: nn.Module, gain: float = 0.01, use_relu_gain: bool = False):
    if isinstance(m, nn.Linear):
        if use_relu_gain:
            gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
