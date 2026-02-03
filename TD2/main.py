import jax
from train import model, collocation_points, t_boundary
from models import u_scalar
from pde import pde_residual
import equinox as eqx
from utils import flatten_grad
import jax.numpy as jnp


grad_wrt_params = eqx.filter_grad(u_scalar)(model, tx=collocation_points[0])
flat_grad = flatten_grad(grad_wrt_params)
print(flat_grad.shape)

grads_wrt_params = jax.vmap(lambda tx: flatten_grad(eqx.filter_grad(u_scalar)(model, tx)))(collocation_points)
print(grads_wrt_params.shape) # Jacobian, shape (N_tx, N_params)
J = grads_wrt_params

K = J @ J.T
print(K.shape) # shape: (N_tx, N_tx)

J_pde = jax.vmap(lambda tx: flatten_grad(eqx.filter_grad(pde_residual)(model, tx)))(collocation_points)
print(J_pde.shape)

left_bc = jnp.stack([t_boundary, jnp.zeros_like(t_boundary)], axis=1)
right_bc = jnp.stack([t_boundary, jnp.ones_like(t_boundary)], axis=1)
print(f"{t_boundary.shape = }")
print(f"{right_bc.shape = }")
boundary_points = jnp.concatenate([left_bc, right_bc], axis=0)
print(f"{boundary_points.shape = }")
J_bc = jax.vmap(lambda tx: flatten_grad(eqx.filter_grad(u_scalar)(model, tx)))(boundary_points)
print(J_bc.shape)


J_full = jnp.concatenate([J_pde, J_bc], axis=0)
K = J_full @ J_full.T
print(K.shape)