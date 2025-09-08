import torch.nn as nn

def make_mlp(input_dim: int, n_classes: int):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, n_classes)
    )
