import torch

def minmax_normalize(x, epsilon=1e-8):
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    x_norm = (x - x_min) / (x_max - x_min + epsilon)
    return x_norm, x_min, x_max

def minmax_denormalize(x_norm, x_min, x_max, epsilon=1e-8):
    return x_norm * (x_max - x_min + epsilon) + x_min
