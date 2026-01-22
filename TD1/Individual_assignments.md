## Individual exercises

**Ex.1** In the heat example that you saw at class, how would you integrate hard constraint on the initial conditions as well?
Make plots that would compare different approaches in terms of convergence and final absolute error.

**Ex.2** Now tackle a new problem. You need to solve Burgers equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2},$$
where $\nu = 0.01/\pi$, with boundary conditions $u(t, -1) = u(t, 1) = 0$ and $u(0,x) = \sin(\pi x)$. The domain is $\Omega = [0, 1] \times [-1, 1]$.

The true solution of this PDE equation is saved in `burgers.mat` file. 

  1. Using example of Heat equation, implement the training of PINN for Burgers, with hard and soft constraints. 
  2. Consider integrating experimental data in the training process. You can use finite difference method to generate the points for a sub-domain $\Omega = [0, \tau] \times [-1, 1]$ with $\tau$ being a parameter to be chosen. Be careful when choosing the time and space steps for FD method so that it remains stable.
  3. Experiment also with a neural network architecture, try different activation functions: relu, gelu, tanh, sin... Compare the performance of different architectures and activation functions in terms of convergence and final absolute error. Can we use non-smooth activation functions? What should be sufficient conditions for a neural network to work well with the PDE problem at hand?

Perform a small research analysis based on three points mentioned above. Understand what helps the training process and achieve the least final absolute error? What did not have any effect or was negligible? When experimenting, design a proper experimental protocol to have the rigorous and fair comparisons.

According to you, what are potential advantages of PINNs compared to traditional numerical methods (rely on the results of your experiments to argument your findings and discuss the results.)

