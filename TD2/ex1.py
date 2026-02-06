import optax
import jax.random as jr
import jax.numpy as jnp
from models import MLP, HardIC
from pde import build_loss, pde_residual
import equinox as eqx
from optax import GradientTransformationExtraArgs
from tqdm import tqdm
from utils import get_A, get_K


# Initilize 
key = jr.PRNGKey(42)
key, model_key, data_key = jr.split(key, 3)
mlp = MLP([2, 32, 32, 32, 1], key=model_key)
ic = lambda x: jnp.sin(jnp.pi * x) # initial condition
hardic = HardIC(mlp, ic)
model = hardic


# create optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


# Sample points
data_key, collocation_key, boundary_key = jr.split(data_key, 3)
collocation_points = jr.uniform(collocation_key, (1000, 2))

# Heat equation: u(t, 0) = 0, u(t, 1) = 0
t_samples = jr.uniform(boundary_key, (100,))
left_pts = jnp.stack([t_samples, jnp.zeros_like(t_samples)], axis=1)
right_pts = jnp.stack([t_samples, jnp.ones_like(t_samples)], axis=1)
boundary_points = jnp.concatenate([left_pts, right_pts], axis=0)
target_values = jnp.zeros(200)  # All zeros for this problem


total_loss = build_loss(pde_residual)


# define training step
@eqx.filter_jit
def train_step(model: HardIC, opt_state, optimizer: GradientTransformationExtraArgs, collocation_points, boundary_points, target_values):
    loss, grads = eqx.filter_value_and_grad(total_loss)(model, collocation_points, boundary_points, target_values)
    updates, opt_state = optimizer.update(grads, opt_state, model) # type: ignore
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


if __name__ == "__main__":
    K_before_train = get_K(model, collocation_points, boundary_points, pde_residual)
    A_before_train = get_A(model, collocation_points, boundary_points, pde_residual)
    losses = []
    for step in tqdm(range(5000)):
        model, opt_state, loss = train_step(model, opt_state, optimizer, collocation_points, boundary_points, target_values)
        losses.append(float(loss))
    K_after_train = get_K(model, collocation_points, boundary_points, pde_residual)
    A_after_train = get_A(model, collocation_points, boundary_points, pde_residual)
        
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
    
    eigenvalues_before = jnp.linalg.eigvalsh(A_before_train) # already sorted in ascending order
    eigenvalues_after = jnp.linalg.eigvalsh(A_after_train) # already sorted in ascending order
    plt.figure()
    plt.plot(eigenvalues_before[::-1], 'o-', label='Before training')
    plt.plot(eigenvalues_after[::-1], 'o-', label='After training')
    plt.yscale('log')
    plt.xlabel('Index (sorted by magnitude)')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.title('Spectrum of A')
    plt.savefig('spectrum_A.png')
    
    kappa_before = eigenvalues_before.max() / eigenvalues_before[eigenvalues_before > 1e-12].min()
    kappa_after = eigenvalues_after.max() / eigenvalues_after[eigenvalues_after > 1e-12].min()
    print(f"Condition number before: {kappa_before:.2e}")
    print(f"Condition number after: {kappa_after:.2e}")


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