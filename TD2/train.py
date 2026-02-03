import optax
from typing import List
from jaxtyping import Float, Array
import jax
import jax.random as jr
import jax.numpy as jnp
from models import MLP, HardIC
from pde import pde_loss
import equinox as eqx
from optax import GradientTransformationExtraArgs
from tqdm import tqdm


# Initilize 
key = jr.PRNGKey(42)
key, model_key, data_key = jr.split(key, 3)
mlp = MLP([2, 32, 32, 32, 1], key=key)
ic = lambda x: jnp.sin(jnp.pi * x)
hardic = HardIC(mlp, ic)
model = hardic

# x = jnp.array([0.5, 0.3])
# output = mlp(x)
# print(output.shape) # (1,)
# print(hardic(jnp.array([0, 0.7]))) # [0.809017]
# print(ic(0.7)) # 0.809017
# print(hardic(jnp.array([0.5, 0.7]))) # [0.42426372]
# residual = pde_residual(hardic, jnp.array([0.5, 0.7]))
# print(residual)

def boundary_loss_ineffecient(model: HardIC, t_points: List[float]):
    """boundary loss for `u(t,0)=0` and `u(t,1)=0`"""
    return sum([(model(jnp.array([t, 0])) ** 2 + model(jnp.array([t, 1])) ** 2) for t in t_points]) / len(t_points)


def boundary_loss(model: HardIC, t_points: Float[Array, "N"]):
    # Create boundary inputs: (N, 2) arrays
    left_bc = jnp.stack([t_points, jnp.zeros_like(t_points)], axis=1) # (t, 0)
    right_bc = jnp.stack([t_points, jnp.ones_like(t_points)], axis=1) # (t, 1)
    
    # Vectorize model over batch dimension
    u_left = jax.vmap(model)(left_bc) # shape: (N, 1)
    u_right = jax.vmap(model)(right_bc) # shape: (N, 1)
    
    return jnp.mean(u_left ** 2 + u_right ** 2)


def total_loss(model: HardIC, collocation_points: Float[Array, "N 2"], t_boundary: Float[Array, "N"], lambda_bc=1.0):
    """ L = L_{PDE} + Lambda * L_{BC} """
    return pde_loss(model, collocation_points) + lambda_bc * boundary_loss(model, t_boundary)


# create optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Sample points
data_key, collocation_key, boundary_key = jr.split(data_key, 3)
collocation_points = jr.uniform(collocation_key, (1000, 2))
t_boundary = jr.uniform(collocation_key, (100, ))

# define training step
@eqx.filter_jit
def train_step(model: HardIC, opt_state, optimizer: GradientTransformationExtraArgs, collocation_points, t_boundary):
    loss, grads = eqx.filter_value_and_grad(total_loss)(model, collocation_points, t_boundary)
    updates, opt_state = optimizer.update(grads, opt_state, model) # type: ignore
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


if __name__ == "__main__":
    losses = []
    for step in tqdm(range(5000)):
        model, opt_state, loss = train_step(model, opt_state, optimizer, collocation_points, t_boundary)
        losses.append(float(loss))
        
    # save the mode
    eqx.tree_serialise_leaves("td2_hardic_model.eqx", model)

    # Plot the loss curve
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("loss.png")


    # import matplotlib.pyplot as plt

    # # Create a grid
    # t_vals = jnp.linspace(0, 1, 50)
    # x_vals = jnp.linspace(0, 1, 50)
    # T, X = jnp.meshgrid(t_vals, x_vals)
    # TX = jnp.stack([T.flatten(), X.flatten()], axis=1)

    # # Predicted solution
    # u_pred = jax.vmap(model)(TX).reshape(50, 50)

    # # Analytical solution
    # u_exact = jnp.exp(-0.1 * jnp.pi**2 * T) * jnp.sin(jnp.pi * X)

    # # Plot
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # im0 = axes[0].contourf(T, X, u_exact, levels=50)
    # axes[0].set_title('Exact')
    # im1 = axes[1].contourf(T, X, u_pred, levels=50)
    # axes[1].set_title('Predicted')
    # im2 = axes[2].contourf(T, X, jnp.abs(u_exact - u_pred), levels=50)
    # axes[2].set_title('Absolute Error')
    # for ax in axes:
    #     ax.set_xlabel('t')
    #     ax.set_ylabel('x')
    # plt.colorbar(im2, ax=axes[2])
    # plt.tight_layout()
    # plt.savefig("error.png")