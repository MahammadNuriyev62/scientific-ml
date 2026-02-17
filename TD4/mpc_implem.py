import numpy as np
import torch
from torchdiffeq import odeint
from scipy.optimize import minimize

UMIN = -2.0
UMAX = 2.0

def integral_cost_numpy(t_grid, cos_theta, omega, tau, w_cos=1.0, w_omega=0.1, w_u=1e-2):
    # Continuous-time objective approximated by trapezoidal integration (NumPy).
    t = np.asarray(t_grid)
    # the cost function represents the distance to the desired equilibrium point + the control effort
    cost = w_cos * (cos_theta - 1.0) ** 2 + w_omega * omega ** 2 + w_u * tau ** 2
    try:
        return np.trapezoid(cost, t)
    except AttributeError:
        return np.trapz(cost, t)

integral_cost = integral_cost_numpy


def _integral_cost_torch(t_grid, cos_theta, omega, tau, w_cos, w_omega, w_u):
    # Torch version of the same integral objective, used inside the optimizer.
    cost = w_cos * (cos_theta - 1.0) ** 2 + w_omega * omega ** 2 + w_u * tau ** 2
    return torch.trapz(cost, t_grid)


class NeuralODEMPC:
    def __init__(
        self,
        model,
        horizon_steps=20,
        dt=0.05,
        u_min=UMIN,
        u_max=UMAX,
        w_cos=1.0,
        w_omega=0.1,
        w_u=1e-2,
        device="cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.w_cos = float(w_cos)
        self.w_omega = float(w_omega)
        self.w_u = float(w_u)
        self.device = device

    def _rollout(self, x0, u_seq):
        # Predict learned dynamics under piecewise-constant torque u_seq.
        # x0: (3,) state [sin, cos, omega]
        # u_seq: (horizon_steps,) control values
        # Return: x_traj of shape (horizon_steps + 1, 3)

        H = len(u_seq)
        # Build a time grid: [0, dt, 2*dt, ..., H*dt]
        t_grid = torch.linspace(0.0, H * self.dt, H + 1, device=self.device)

        # Build piecewise-constant control: at each grid point, use the
        # control for that interval (last point repeats the final control)
        tau_pc = torch.cat([u_seq, u_seq[-1:]], dim=0)  # (H+1,)
        tau_pc = tau_pc.unsqueeze(0)  # (1, H+1) batch dim

        self.model.set_control(t_grid, tau_pc)

        # Integrate in a single call with euler (fast for MPC optimization)
        x = x0.unsqueeze(0)  # (1, 3)
        x_traj = odeint(self.model, x, t_grid, method='euler')  # (H+1, 1, 3)
        x_traj = x_traj.squeeze(1)  # (H+1, 3)
        return x_traj

    def _cost(self, u_np, x0_np, t_grid = None):
        # MPC objective = integral of state + control cost over the prediction horizon.
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        # here u_seq is the control sequence, i.e. torques
        u_seq = torch.tensor(u_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            x_traj = self._rollout(x0, u_seq)

        cos_theta = x_traj[:, 1]
        omega = x_traj[:, 2]
        # Repeat the last control to compute the cost of the last interval in integral cost
        tau_traj = torch.cat([u_seq, u_seq[-1:]], dim=0)
        if t_grid is None:
            t_grid = torch.linspace(
                0.0, self.horizon_steps * self.dt, self.horizon_steps + 1, device=self.device
            )

        total = _integral_cost_torch(
            t_grid, cos_theta, omega, tau_traj, self.w_cos, self.w_omega, self.w_u
        )
        return total.item()

    def solve(self, x0_np, u_init=None, maxiter=50, ftol=1e-7):
        # Optimize a control sequence for the current state.
        if u_init is None:
            u_init = np.zeros(self.horizon_steps, dtype=np.float32)
        bounds = [(self.u_min, self.u_max)] * self.horizon_steps

        res = minimize(
            self._cost,
            u_init,
            args=(x0_np,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(ftol), "eps": 1e-3},
        )
        return res.x

    def step(self, x0_np, u_prev=None, maxiter=50, ftol=1e-7):
        # Compute the optimal control for the current state and return the first control input.
        if u_prev is None:
            u_prev = np.zeros(self.horizon_steps, dtype=np.float32)
        u_seq = self.solve(x0_np, u_init=u_prev, maxiter=maxiter, ftol=ftol)
        u0 = float(u_seq[0])
        return u0, u_seq


def simulate_mpc(model, x0_np, steps=100, **mpc_kwargs):
    # Roll out the learned model in closed loop for diagnostics.
    mpc = NeuralODEMPC(model, **mpc_kwargs)
    x0_np = np.array(x0_np, dtype=np.float32)

    if x0_np.ndim == 1:
        x = x0_np
        traj = [x.copy()]
        u_applied = []
        u_prev = np.zeros(mpc.horizon_steps, dtype=np.float32)

        for _ in range(int(steps)):
            u0, u_seq = mpc.step(x, u_prev=u_prev)
            u_applied.append(u0)

            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32, device=mpc.device)
                u_t = torch.tensor([u0], dtype=torch.float32, device=mpc.device)
                x_next = mpc._rollout(x_t, u_t)[1].cpu().numpy()

            x = x_next
            traj.append(x.copy())

            u_prev = np.roll(u_seq, -1)
            u_prev[-1] = 0.0

        return np.array(traj), np.array(u_applied)

    trajs = []
    u_trajs = []
    for b in range(x0_np.shape[0]):
        traj_b, u_b = simulate_mpc(model, x0_np[b], steps=steps, **mpc_kwargs)
        trajs.append(traj_b)
        u_trajs.append(u_b)

    return np.stack(trajs, axis=0), np.stack(u_trajs, axis=0)



def run_single_mpc_closed_loop(
    model,
    y0_single,
    steps=100,
    dt=0.05,
    u_min=UMIN,
    u_max=UMAX,
    mpc_kwargs=None,
):
    # Closed-loop MPC on true dynamics: learned model plans, true dynamics executes.
    # y0_single: (2,) array [theta, omega]
    # Returns: traj (steps+1, 2) from true dynamics, u_applied (steps,) controls
    from pendulum_gp_dataset import rollout_closed_loop

    if mpc_kwargs is None:
        mpc_kwargs = {}
    mpc = NeuralODEMPC(model, dt=dt, u_min=u_min, u_max=u_max, **mpc_kwargs)

    y = np.array(y0_single, dtype=np.float32)  # [theta, omega]
    traj = [y.copy()]
    u_applied = []
    u_prev = np.zeros(mpc.horizon_steps, dtype=np.float32)

    for _ in range(int(steps)):
        # Convert true state (theta, omega) to Neural ODE state (sin, cos, omega) for planning
        theta_curr, omega_curr = float(y[0]), float(y[1])
        x_sincos = np.array([np.sin(theta_curr), np.cos(theta_curr), omega_curr], dtype=np.float32)

        # Plan using the learned model
        u0, u_seq = mpc.step(x_sincos, u_prev=u_prev)
        u_applied.append(u0)

        # Execute one step on the true dynamics
        t_step = torch.tensor([0.0, dt], dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, 2)

        def control_fn(t, state, _u0=u0):
            return torch.full((state.shape[0],), _u0)

        y_next = rollout_closed_loop(y_torch, t_step, control_fn)  # (1, 2, 2)
        y = y_next[0, -1, :].numpy().copy()  # final state (theta, omega)
        traj.append(y.copy())

        # Warm-start: shift the previous solution
        u_prev = np.roll(u_seq, -1)
        u_prev[-1] = 0.0

    traj = np.array(traj)  # (steps+1, 2)
    return traj, np.array(u_applied, dtype=np.float32)
