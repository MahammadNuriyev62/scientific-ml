
from typing import Callable, List, cast
from models import HardIC
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp


def u(model: HardIC, tx: Float[Array, "2"]):
    """Scalar-valued version of model output."""
    return model(tx)[0]


grad_u = jax.grad(u, argnums=1)


def du_dt(model: HardIC, tx: Float[Array, "2"]):
    return cast(float, grad_u(model, tx)[0])


def du_dx(model: HardIC, tx: Float[Array, "2"]):
    return cast(float, grad_u(model, tx)[1])


def d2u_dx2(model: HardIC, tx: Float[Array, "2"]):
    return cast(float, jax.grad(du_dx, argnums=1)(model, tx)[1])


PDEResidual = Callable[[HardIC, Array], float]

def pde_residual(model: HardIC, tx: Float[Array, "2"]):
    return du_dt(model, tx) - 0.1 * d2u_dx2(model, tx)


def boundary_loss_ineffecient(model: HardIC, t_points: List[float]):
    """boundary loss for `u(t,0)=0` and `u(t,1)=0`"""
    return sum([(model(jnp.array([t, 0])) ** 2 + model(jnp.array([t, 1])) ** 2) for t in t_points]) / len(t_points)


def build_loss(pde_residual: PDEResidual):
    def pde_loss(model, txs: Float[Array, "N 2"]):
        pde_residuals = jax.vmap(lambda tx: pde_residual(model, tx))(txs)
        return jnp.mean(pde_residuals ** 2)    
    
    def boundary_loss(model: HardIC, boundary_points: Float[Array, "N 2"], target_values: Float[Array, "N"]):
        u_pred = jax.vmap(model)(boundary_points)
        return jnp.mean((u_pred.squeeze() - target_values) ** 2)
    
    def total_loss(model: HardIC, collocation_points: Float[Array, "N 2"], boundary_points: Float[Array, "N 2"], target_values: Float[Array, "N"], lambda_bc=1.0):
        """ L = L_{PDE} + Lambda * L_{BC} """
        return pde_loss(model, collocation_points) + lambda_bc * boundary_loss(model, boundary_points, target_values)
    
    return total_loss