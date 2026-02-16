import math
import torch
import gpytorch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

UMAX = 2.0
UMIN = -2.0



# helping functions to move between sin/cos and theta representations
def to_sincos(theta):
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def to_theta(sin_theta, cos_theta):
    return torch.atan2(sin_theta, cos_theta)


def sample_gp_controls(t, batch, lengthscale=0.3, outputscale=1.0, noise=1e-3, device="cpu", dtype=torch.float32):
    """Sample tau'(t) from a zero-mean GP on time grid t. Returns (batch, T)."""
    t = t.to(device=device, dtype=dtype)
    T = t.numel()
    # GP kernel
    base_kernel = gpytorch.kernels.RBFKernel()
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    base_kernel.lengthscale = torch.as_tensor(lengthscale, device=device, dtype=dtype)
    kernel.outputscale = torch.as_tensor(outputscale, device=device, dtype=dtype)

    t_col = t.view(-1, 1)
    cov = kernel(t_col).evaluate()  # (T, T)
    cov = cov + noise * torch.eye(T, device=device, dtype=dtype)
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(T, device=device, dtype=dtype),
        covariance_matrix=cov,
    )
    samples = mvn.sample((batch,))  # (batch, T)
    return samples.clamp(UMIN, UMAX)


class PendulumDynamics(torch.nn.Module):
    def __init__(self, t_grid=None, tau_grid=None, control_fn=None, g=9.81, L=1.0):
        super().__init__()
        if t_grid is not None:
            self.register_buffer("t_grid", t_grid)
        else:
            self.t_grid = None
        if tau_grid is not None:
            self.register_buffer("tau_grid", tau_grid)
        else:
            self.tau_grid = None
        self.control_fn = control_fn
        self.g = g
        self.L = L

    def _tau_at(self, t):
        # Linear interpolation on the time grid for each batch
        t_grid = self.t_grid
        tau = self.tau_grid
        # indices of the right neighbor
        idx = torch.searchsorted(t_grid, t).clamp(1, t_grid.numel() - 1)
        t0 = t_grid[idx - 1]
        t1 = t_grid[idx]
        w = (t - t0) / (t1 - t0 + 1e-12)
        tau0 = tau[:, idx - 1]
        tau1 = tau[:, idx]
        return tau0 + w * (tau1 - tau0)

    def forward(self, t, state):
        theta = state[:, 0]
        omega = state[:, 1]
        if self.control_fn is not None:
            tau_t = self.control_fn(t, state)
        else:
            tau_t = self._tau_at(t)
        dtheta = omega
        domega = (self.g / self.L) * torch.sin(theta) + tau_t
        return torch.stack([dtheta, domega], dim=-1)


class PendulumDatasetGenerator:
    def __init__(self, t0=0.0, t1=5.0, n_timesteps=200, g=9.81, L=1.0, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.t = torch.linspace(t0, t1, n_timesteps, device=device, dtype=dtype)
        self.g = g
        self.L = L

    def generate(self, n_sims=16, delta=1., gp_lengthscale=0.3):
        # Initial conditions around upright position (pi)
        theta0 = (math.pi - (2 * torch.rand(n_sims, device=self.device, dtype=self.dtype) - 1) * delta)
        omega0 = torch.randn(n_sims, device=self.device, dtype=self.dtype)
        y0 = torch.stack([theta0, omega0], dim=-1)

        tau = sample_gp_controls(self.t, n_sims, lengthscale=gp_lengthscale, device=self.device, dtype=self.dtype)
        dyn = PendulumDynamics(self.t, tau, g=self.g, L=self.L).to(self.device)

        traj = odeint(dyn, y0, self.t, method="dopri5")  # (T, batch, 2)
        traj = traj.permute(1, 0, 2).contiguous()  # (batch, T, 2)

        # Output formatting: (t, theta, sin(theta), cos(theta), omega, tau') as tensors
        t = self.t.expand(n_sims, -1)
        theta = traj[:, :, 0]
        omega = traj[:, :, 1]
        theta_sin = torch.sin(theta)
        theta_cos = torch.cos(theta)
        return t, theta, theta_sin, theta_cos, omega, tau

# the same as above but with an abstract control function
def rollout_closed_loop(
    y0,
    t_grid,
    control_fn,
    g=9.81,
    L=1.0,
    method="dopri5",
):
    dyn = PendulumDynamics(t_grid=None, tau_grid=None, control_fn=control_fn, g=g, L=L)
    traj = odeint(dyn, y0, t_grid, method=method)  # (T, batch, 2)
    return traj.permute(1, 0, 2).contiguous()


def plot_example(t, theta, omega, tau, idx=0, out_path=None):
    t_np = t[idx].cpu().numpy()
    theta_np = theta[idx].cpu().numpy()
    omega_np = omega[idx].cpu().numpy()
    tau_np = tau[idx].cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    axs[0].plot(t_np, theta_np)
    axs[0].set_ylabel("theta")
    axs[1].plot(t_np, omega_np)
    axs[1].set_ylabel("omega")
    axs[2].plot(t_np, tau_np)
    axs[2].set_ylabel("tau'")
    axs[2].set_xlabel("t")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    gen = PendulumDatasetGenerator(t0=0.0, t1=5.0, n_timesteps=200)
    t, theta, theta_sin, theta_cos, omega, tau = gen.generate(n_sims=8, delta=0.2, gp_lengthscale=0.4)
    print(
        "t shape:", t.shape,
        "theta shape:", theta.shape,
        "sin shape:", theta_sin.shape,
        "cos shape:", theta_cos.shape,
        "omega shape:", omega.shape,
        "tau shape:", tau.shape,
    )
    plot_example(t, theta, omega, tau, idx=0)
