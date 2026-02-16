# Individual Assignment: Neural ODE Control for Pendulum Dynamics

## Objective
Implement a complete pipeline for learning and controlling pendulum dynamics using Neural ODEs and Model Predictive Control (MPC).

## Background

### Inverted Pendulum Dynamics

The inverted pendulum is a classic control problem in which a pendulum is attached to a motorized cart or pivot point, with the pole initially pointing upwards. The goal is to keep the pendulum balanced in the upright (inverted) position, which is unstable, through active control.

**System Description:**
- The pendulum has mass $m$ and length $l$, with angle $\theta$ measured from the vertical (upright) position
- The system is subject to gravity and friction
- A control input $u$ (torque or force) can be applied at the pivot to influence the dynamics

**Dynamics Equation:**
The continuous-time dynamics of the inverted pendulum are governed by:

$$\ddot{\theta} = \frac{g}{l} \sin(\theta) + \tau$$

where:
- $\theta$ is the angle from vertical (upright)
- $\dot{\theta}$ is the angular velocity
- $\ddot{\theta}$ is the angular acceleration
- $g$ is gravitational acceleration
- $\tau$ is the control input (torque), further can be also denoted as $u$ (control)
- $\tau$ is bounded by $|\tau| \leq \tau_{max}$, here $\tau_{max} = 2.0$
- $l$ is the length of the pendulum

**Key Properties:**
- The upright position ($\theta = 0$) is unstable: small disturbances grow exponentially without control
- To stabilize this system, we need active feedback control that responds to the current state


### Model Predictive Control (MPC)

Model Predictive Control is an optimization-based control strategy that uses a learned or known dynamical model to predict future system behavior and compute optimal control actions.

**How MPC Works:**
1. **Prediction Horizon**: MPC considers predictions over a future time window (horizon) of length $T$
2. **Optimization**: At each time step, solve an optimization problem to find the sequence of control inputs $u_0, u_1, \ldots, u_{T-1}$ that minimizes a cost function while respecting constraints
3. **Receding Horizon**: Apply only the first control action $u_0$, then shift forward in time and repeat

**Optimization Objective:**
Minimize the cost:

$$J = \sum_{t=0}^{T-1} \left( \|x_t - x_{ref}\|^2 + \lambda \|u_t\|^2 \right)$$

where:
- $x_t$ is the predicted state at time $t$
- $x_{ref}$ is the reference (target) state (e.g., upright position)
- $u_t$ is the control input at time $t$
- $\lambda$ is a regularization parameter balancing state tracking and control effort

This objective function can be solved using quadratic programming (QP), for instance. In this assignment, we will use `scipy.optimize.minimize` to optimize the cost function with respect to the control inputs $u_t$ for $t = 0, \ldots, T-1$, under the constraints $u_{min} \leq u_t \leq u_{max}$ (in case of the pendulum $u$ is a torque $\tau$).

**In This Assignment:**
You will use the learned Neural ODE model as the prediction model for MPC. The neural network learns the pendulum dynamics from data, and then MPC uses these learned dynamics to compute control inputs that stabilize the pendulum.

## Tasks

**Q1. Neural ODE Training**
- Generate a dataset of pendulum trajectories using the true dynamics generated from `pendulum_gp_dataset.py`
- Implement a Neural ODE model and train it using the generated dataset until convergence, use $\sin \theta$ and $\cos \theta$ to describe the angle
- Evaluate training performance (e.g., trajectory prediction error, don't forget to divide your dataset into train and test sets)

**Q2. Model Predictive Control**
- See `mpc_implem.py` 
- Implement an MPC controller using the trained Neural ODE as the learned dynamics model (Implement #TODO comments)
- Implement the function `run_single_mpc_closed_loop` in `mpc_implem.py` that generate a single trajectory using the learned model and applies the control to the true dynamics, you can use `rollout_closed_loop` function from `pendulum_gp_dataset.py`


**Q.3. Comparison with True Dynamics**
- Deploy the MPC controller on the **true pendulum dynamics** (not the learned model)
- Run complete rollout trajectories and record the integral cost (see the function `integral_cost_numpy` in `mpc_implem.py`)
- Compare the integral cost obtained with MPC with the integral cost computed on trained trajectories (see the function `integral_cost_numpy` in `mpc_implem.py`)

**Q.4. Improving MPC Performance (Bonus)**
Analyse your finding and propose an approach to improve the performance of the MPC controller, implement your approach, can you achieve the best performance? 

*Hint:* plot the trajectories for $\cos \theta$, do you see stabilization at the target position? Experiment with the choice of integration function during Neural ODE training and MPC optimization, and the choice of MPC horizon.

