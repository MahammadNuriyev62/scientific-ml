## Individual exercises


**Ex.0** Look at Poisson eqation, i.e. $\frac{\partial^2 u}{\partial x^2} = 0$, with boundary conditions $u(0) = u(1) = 0$. Assume you have fourier features and your machine learning model is learning the weights of each fourier feature. The basis functions are  $\phi_0(x) = \frac1{\sqrt{2\pi}}$, $\phi_{-k}(x) =\frac1{\sqrt{\pi}}\cos(kx)$ and $\phi_{k}(x) =\frac1{\sqrt{\pi}}\sin(kx)$, and your model is $u(x) = \sum_{k=-K}^K a_k \phi_k(x)$.
Analyse how the conditioning number of matrix A (see lecture notes) depends on the $k$, i.e. a frequency of the basis function.


**Practical Assignments**


Let's consider Heat and Burgers equations again. Reminder, the PDE for Heat equation

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},$$
where $\alpha = 0.1$, with boundary conditions $u(0, t) = u( 1,t) = 0$ and $u(x, 0) = \sin(\pi x)$. The domain is $\Omega = [0, 1] \times [0, 1]$.

And for Burgers equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2},$$
where $\nu = 0.01/\pi$, with boundary conditions $u(t, -1) = u(t, 1) = 0$ and $u(0,x) = \sin(\pi x)$. The domain is $\Omega = [0, 1] \times [-1, 1]$.

**Ex.1** (a) Implement the empirical Neural Tangent Kernel (NTK) calculation, i.e. the NTK matrix consists of the following entries $K = \begin{bmatrix} \Theta_{pp} & \Theta_{pb} \\\
\Theta_{bp} & \Theta_{bb} \end{bmatrix} \in \mathbb{R}^{N_f+N_b \times N_f+N_b}$, where $N_f$ is the number of collocation points in the domain $\Omega$ and $N_b$ is the number of collocation points on the boundary $\partial \Omega$, and $\Theta_{pp, ij} = \nabla_\theta \mathcal{N}  u_\theta(x_i) \nabla_\theta \mathcal{N} u_\theta(x_j)^T$, for $x_i, x_j \in \Omega$, $\Theta_{pb, ij} = \nabla_\theta \mathcal{N}  u_\theta(x_i) \nabla_\theta \mathcal{B} u_\theta(x_j)^T$, for $x_i \in \Omega$ and $x_j \in \partial \Omega$, $\Theta_{bp, ij} = \nabla_\theta \mathcal{B}  u_\theta(x_i) \nabla_\theta \mathcal{N} u_\theta(x_j)^T$, for $x_i \in \partial \Omega$ and $x_j \in \Omega$, and $\Theta_{bb, ij} = \nabla_\theta \mathcal{B}  u_\theta(x_i) \nabla_\theta \mathcal{B} u_\theta(x_j)^T$, for $x_i, x_j \in \partial \Omega$. 
Run an Adam based training with soft constraints for boundary conditions and hard constraint for initial conditions for test equations, therefore $u_\theta$ in NTK definition is after hard constraint postprocessing, plot the NTK maps, to see an influence of a few training points on the whole domain, measured at the beginning and at the end of the training process, for instance select 10 random points from the training dataset and for each selected point $(t_i,x_i)$, plot matrix $K((t_i,x_i), X_{grid})$, where $X_{grid}$ is the dense grid of points in the domain $\Omega$.

(b) Compute matrix A from the lecture notes and plot its spectrum.

(c) Try random weight factorization instead of the standard neural network. Redo the experiments from (a) and (b). Do you see any difference?

*Tip:* Don't use big dataset to make NTK calculation feasible. Implement the NTK calculation in the form of $J J^T$, where J is the concatenation of the Jacobians of the PDE and boundary conditions. Use the functions from `utils.py` to flatten (represent all the parameters as one big vector) and unflatten the parameters (from the vector of parameters retrieve the original shape of parameters). Similarly for matrix A, observe that to compute A empirically, you can reuse the Jacobians of the PDE and boundary conditions and compute it as $J^T J$.


**Ex.2** Try now a quasi-Newton type of optimizers. Use L-BFGS optimizer to train PINNs for the test equations. Compare the results with Adam optimizer.

*Tip:* You can use Adam optimizer at the beginning of the training, for a few iterations, to get a good initial guess for L-BFGS. So that the method converges faster.

