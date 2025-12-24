# Domain Study: Guide

To do meaningful research in a domain, you need to learn what others already do
and don't understand in this area. Use this folder to organize your group's
understanding of your research domain including: your own summaries, helpful
PDFs, links you found helpful, ...

This folder is different from `/notes` because it contains _only_ information
about your research domain. When deciding what goes here, ask yourself this
question: _Would someone need to know this to understand our research?_

## README.md

Use this folder's README to document all the notes and resources in this folder.
Someone shouldn't need to read through _everything_ to find what they need.

## Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are a class of neural networks that incorporate governing physical laws (typically expressed as differential equations) directly into their training objective. In practice, a PINN is trained not only to fit any available data, but also to minimize the residual of the known physical equations (ordinary or partial differential equations) that describe the system. This means the loss function of a PINN includes terms for the equation residual (physics loss) in addition to the usual data mismatch loss. By doing so, PINNs leverage prior knowledge of physics to guide learning, which tends to improve generalization and accuracy even with sparse or noisy data. Unlike traditional purely data-driven models, PINNs can enforce physical constraints (such as conservation laws or boundary conditions) and thus predict physically consistent outcomes beyond the training range. They are also mesh-free and can handle high-dimensional problems, offering an alternative to classical solvers for forward and inverse problems in science and engineering.

### Benefits of PINNs

In contexts where data is limited or expensive to obtain, PINNs can embed the essential physics to reduce reliance on large datasets. They excel at solving both forward problems (finding system states given parameters) and inverse problems (inferring unknown parameters or inputs from observed states) by naturally integrating physical equations into the learning process. For example, PINNs have been used to infer material properties (like thermal conductivity or elasticity constants) by fitting observations while obeying the governing PDEs. Compared to traditional numerical solvers, PINNs are mesh-free and can handle irregular geometries or higher-dimensional spaces more easily. They also allow combining data with physical laws seamlessly – e.g., one can enforce partial differential equations and boundary conditions in the loss function while also fitting any available sensor data, which is particularly useful for digital twin models and systems where some physics is known and some must be learned from data.

Training a PINN involves an optimization (usually gradient-based) that adjusts the network weights to satisfy both data fidelity and physics fidelity. The physics equations are imposed by evaluating the network’s outputs through automatic differentiation to compute the differential equation residuals. These residuals are added to the loss, effectively penalizing the network when it produces outputs that violate the physical laws. Boundary conditions and initial conditions can be enforced either by adding corresponding penalty terms (soft constraints) or by designing the network architecture to inherently satisfy them (hard constraints). The balance between the physics loss and data loss is crucial – in practice one may weight these terms to ensure neither the data fit nor the physics constraints dominate improperly. PINNs thus perform a form of multi-task learning, where the "tasks" are fitting the data and satisfying the differential equations simultaneously. This synergy enables PINNs to predict behaviors outside the range of training data by relying on physical principles, something standard black-box models struggle with.

### Limitations and Challenges of PINNs

One open challenge in the field is enabling PINNs to learn chaotic or strongly nonlinear dynamics in a faithful and stable manner. Standard PINNs often fail in chaotic regimes because training is framed as a global optimization problem that does not respect temporal causality, allowing errors to propagate backward and forward in time without constraint. As a result, the network may converge to spurious trajectories or diverge entirely when long-time integration or sensitive dependence on initial conditions is involved. Driven nonlinear oscillators such as the Duffing system, particularly in chaotic regimes, have therefore emerged as stringent test cases. To date, there are relatively few demonstrations of PINNs successfully reproducing chaotic attractors, and those that exist rely on modified training strategies, indicating that this remains an open and active area of research rather than a solved problem.

One class of approaches proposed to address this limitation focuses on enforcing temporal causality during training. Causality-aware PINNs reformulate the loss function to respect the chronological structure of time evolution, for example by weighting residuals so that earlier time errors are resolved before later ones. Wang et al. (2022) demonstrated that without such causality-preserving mechanisms, PINNs fail to reproduce chaotic systems such as the Lorenz attractor, while modified formulations significantly improve stability. Related strategies include domain decomposition or sequential learning, in which the time interval is split into smaller subdomains and the solution is trained incrementally rather than over the full horizon at once. This effectively transforms a long, unstable optimization into a sequence of shorter, better-conditioned problems and has been shown to mitigate error accumulation in highly oscillatory or nonlinear systems.

Another promising but still underdeveloped direction involves incorporating known physical invariants into the PINN loss function. Prior work has shown that adding an energy-based regularization term can improve convergence and stability for systems such as the Duffing oscillator, particularly when data is sparse. However, there is no general framework for incorporating multiple invariants, handling approximate invariants, or adapting invariant weights during training. Open questions include how to combine several conserved or slowly varying quantities, how invariant constraints behave in dissipative systems where conservation laws break down, and whether such constraints improve performance only in low-data regimes or also enhance accuracy and robustness when data is abundant. These issues place invariant-informed PINNs at the boundary between principled physics integration and heuristic regularization.

Loss balancing remains a further unresolved challenge. PINNs rely on multiple competing objectives—data fidelity, physics residuals, boundary conditions, and sometimes invariant constraints—and the relative weighting of these terms strongly influences convergence. In practice, weights are often chosen manually or tuned heuristically, with no guarantee of optimality. Adaptive loss balancing methods, including schemes based on gradient norms or dynamic rescaling during training, have been proposed but are not yet standardized or theoretically well understood. Poorly chosen weights can cause the network to satisfy the governing equations while ignoring data, or to fit data while violating physical laws, especially in stiff or chaotic systems.

Taken together, these challenges clarify why applying PINNs to long-time, chaotic, or highly nonlinear dynamical systems remains difficult despite their conceptual appeal. They also explain why benchmark systems such as nonlinear oscillators continue to serve as important testbeds for evaluating new training strategies, architectural constraints, and physics-informed regularization techniques.

### Future Directions

PINNs are a dynamic and developing field, with ongoing research exploring improved training strategies, adaptability in weighting loss components, and innovative approaches to handle their limitations. Investigating modifications or enhancing capabilities for chaotic systems and long-time horizon problems are particularly active research areas.

### References

1. [What Are Physics-Informed Neural Networks - MathWorks](https://www.mathworks.com/discovery/physics-informed-neural-networks.html#:~:text=What%20Are%20Physics,PINNs)
2. [Resistance Condition of PINNs - AIMS Press](https://www.aimspress.com/article/doi/10.3934/math.2025364#:~:text=resistance%20condition,the%20minimizer%20of%20the%20loss)
3. [Wave Equation in PINNs - MDPI](https://www.mdpi.com/1424-8220/23/5/2792#:~:text=The%20wave%20equation%20has%20been,on%20the%20propagation%20of%20a)
4. [Van der Pol and Chaotic Dynamics - ArXiv](https://arxiv.org/html/2408.11077v1#:~:text=Van%20der%20Pol%2C%20and%20Duffing,and%20can%20even%20predict%20solutions)
5. [Loss Function Challenge in PINNs - ArXiv](https://arxiv.org/abs/2203.07404#:~:text=,loss%20functions%20that%20can%20explicitly)
6. [Ensemble PINNs for Inverse Transport - Bohrium](https://www.bohrium.com/paper-details/ensemble-physics-informed-neural-networks-a-framework-to-improve-inverse-transport-modeling-in-heterogeneous-domains/875705581518717745-3872#:~:text=informed%20neural%20networks%20,set%20of%20patterns%20can%20guide)
