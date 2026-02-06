
import numpy as np
from matplotlib import pyplot as plt

fish_interior = np.load("fish_interior.npy")
fish_boundary = np.load("fish_boundary.npy")
fish_dist = np.load("fish_dist.npy")


def load_data(interior_path, boundary_path, dist_path):
    data_interior = np.load(interior_path)
    data_boundary = np.load(boundary_path)
    data_dist = np.load(dist_path) 
    if data_dist.ndim > 1:
        data_dist = data_dist.reshape(-1)
    return data_interior, data_boundary, data_dist

def make_scaler(points):
    center = points.mean(axis=0)
    span = points.max(axis=0) - points.min(axis=0)
    scale = float(span.max())
    return center, scale

def apply_scaler(points, center, scale):
    return (points - center) / scale

def normalize_data(data_interior, data_boundary, data_dist):
    all_points = np.concatenate([data_interior, data_boundary], axis=0)
    center, scale = make_scaler(all_points)
    data_interior_n = apply_scaler(data_interior, center, scale)
    data_boundary_n = apply_scaler(data_boundary, center, scale)
    data_dist_n = data_dist / scale
    return data_interior_n, data_boundary_n, data_dist_n

def plot_data(data_interior, data_boundary):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data_interior[:, 0], data_interior[:, 1], s=1, alpha=0.4, label="interior")
    ax.scatter(data_boundary[:, 0], data_boundary[:, 1], s=2, alpha=0.8, label="boundary")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Problem geometry (normalized)")
    plt.show()


