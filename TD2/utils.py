import equinox as eqx
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import jax.flatten_util
from jaxtyping import Float, Array
from pde import PDEResidual

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


def get_J(model, collocation_points: Float[Array, "N 2"], boundary_points: Float[Array, "N 2"], pde_residual: PDEResidual):
    def u_scalar(model, tx):
        return model(tx)[0]
    
    J_pde = jax.vmap(lambda tx: flatten_grad(eqx.filter_grad(pde_residual)(model, tx)))(collocation_points)
    J_bc = jax.vmap(lambda tx: flatten_grad(eqx.filter_grad(u_scalar)(model, tx)))(boundary_points)
    J_full = jnp.concatenate([J_pde, J_bc], axis=0)
    
    return J_full, J_pde, J_bc


def get_K(model, collocation_points: Float[Array, "N 2"], boundary_points: Float[Array, "N 2"], pde_residual: PDEResidual):
    J_full = get_J(model, collocation_points, boundary_points, pde_residual)[0] # (n_data, n_params)
    K = J_full @ J_full.T # (n_data, n_data)
    return K
    
    
def get_A(model, collocation_points: Float[Array, "N 2"], boundary_points: Float[Array, "N 2"], pde_residual: PDEResidual):
    J_full = get_J(model, collocation_points, boundary_points, pde_residual)[0] # (n_data, n_params)
    A = J_full.T @ J_full # (n_params, n_params)
    return A