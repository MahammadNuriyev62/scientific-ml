
from models import HardIC
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp


def u(model: HardIC, tx: Float[Array, "2"]):
    """Scalar-valued version of model output."""
    return model(tx)[0]


grad_u = jax.grad(u, argnums=1)


def du_dt(model: HardIC, tx: Float[Array, "2"]):
    return grad_u(model, tx)[0]


def du_dx(model: HardIC, tx: Float[Array, "2"]):
    return grad_u(model, tx)[1]


def d2u_dx2(model: HardIC, tx: Float[Array, "2"]):
    return jax.grad(du_dx, argnums=1)(model, tx)[1]


def pde_residual(model: HardIC, tx: Float[Array, "2"]):
    return du_dt(model, tx) - 0.1 * d2u_dx2(model, tx)


def pde_loss(model, txs: Float[Array, "N 2"]):
    pde_residuals = jax.vmap(lambda tx: pde_residual(model, tx))(txs)
    return jnp.mean(pde_residuals ** 2)

