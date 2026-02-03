import equinox as eqx
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import jax.flatten_util

@dataclass
class FlatParams:
    flat: jnp.ndarray
    unravel: callable


def flatten_params(model) -> FlatParams:
    params = eqx.filter(model, eqx.is_array)
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    return FlatParams(flat=flat, unravel=unravel)


def replace_params(model, flat: FlatParams, vec: jnp.ndarray):
    params = flat.unravel(vec)
    return eqx.combine(params, eqx.filter(model, eqx.is_array, inverse=True))


def flatten_grad(grad_pytree):
    """Flatten a gradient pytree into a single vector."""
    grad_arrays = eqx.filter(grad_pytree, eqx.is_array)
    flat, _ = jax.flatten_util.ravel_pytree(grad_arrays)
    return flat


def get_JK(model, collocation_points, t_boundary):
    ...