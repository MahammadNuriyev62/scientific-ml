
import jax
import jax.numpy as jnp
import jax.random as jrandom

NU = 0.01 / jnp.pi

def initial_condition(x):
    return -jnp.sin(jnp.pi * x)

def finite_difference_reference(nx=128, nt=200, t_min = 0.0, t_max=1.0, x_min=-1.0, x_max=1.0, t_grid=None):
    """Explicit FD solver; use only for small t. t_grid can override time stamps."""
    xs = jnp.linspace(x_min, x_max, nx)
    if t_grid is not None:
        ts = jnp.asarray(t_grid)
        nt = ts.shape[0]
    else:
        ts = jnp.linspace(t_min, t_max, nt)
    dx = xs[1] - xs[0]
    dt = ts[1] - ts[0]
    u = jnp.zeros((nt, nx))
    u = u.at[0].set(initial_condition(xs))
    for k in range(nt - 1):
        u_k = u[k]
        u_x = (jnp.roll(u_k, -1) - jnp.roll(u_k, 1)) / (2 * dx)
        u_xx = (jnp.roll(u_k, -1) - 2 * u_k + jnp.roll(u_k, 1)) / (dx ** 2)
        u_next = u_k - dt * (u_k * u_x) + dt * NU * u_xx
        u_next = u_next.at[0].set(0.0).at[-1].set(0.0)
        u = u.at[k + 1].set(u_next)
    return xs, ts, u

