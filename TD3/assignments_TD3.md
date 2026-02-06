Consider Eikonal equation in 2D under different geometries. 

Reminder of Eikonal equation:
$$ \|\nabla u\|^2  - 1 = 0$$ 
and its viscsosity regularized equation: 
$$\|\nabla u\|^2  - 1 = \epsilon \Delta u,$$ 
where $\epsilon$ is a small positive number and $\Delta = \sum_{i=1}^2 \frac{\partial^2}{\partial x_i^2}$ is the Laplacian operator. We consider the Dirichlet boundary condition $u(x) = 0$ on the boundary of the domain.

The true solution of the Eikonal equation is the distance function to the boundary of the domain.

**Q1** Consider the simple case of a hypercube, i.e. $x \in [-1,1] \times [-1,1]$. Find the solution of the Eikonal equation using PINNs. You should approximate as closely as possible the true solution of the equation under $\epsilon = 0.$, i.e. solve the unregularized equation.

**Q2** Consider a more complex geometry, e.g. a circle with a square hole. Redo the same task as in Q1.

**Q3** Finally, consider a non-trivial geometry. You have access to files `fish_interior.npy`, `fish_boundary.npy`, and `fish_dist.npy` that contain the already sampled points of the interior, of the boundary and the true disntances of interior points to the boundary. Use these points to train a PINN to solve Eikonal equation. You can use `data_utils.py` to load, rescale and visualize the data.

*Tips:* Note that the Vanilla PINN approach may fail for this problem. Use different tricks to improve the training, for example, curriculum learning by warmstarting the training with a larger $\epsilon$ and then gradually decreasing it to values close to zero; use hard boundary conditions where it is possible; use self-adaptive weighting of the loss terms to make sure that the boundary condition is well satisfied; use adaptive sampling strategies to focus on the regions where the error is large, which can be especially useful for complex geometries; feel free to use any other tricks you find in the literature.