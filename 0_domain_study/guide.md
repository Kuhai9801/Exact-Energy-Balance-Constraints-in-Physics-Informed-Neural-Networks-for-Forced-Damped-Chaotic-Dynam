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

Despite the successes mentioned, PINNs come with several limitations and known challenges, especially when applied to complex or long-time dynamical systems. Understanding these issues is crucial, as they point to why certain problems (like our chosen system) are not “solved” yet and what pitfalls to avoid or investigate. Key challenges include:

- *Convergence and Initialization Issues*: PINNs often require careful tuning to converge to the correct solution. Unlike classical solvers with guaranteed convergence (under certain conditions), PINN training is an optimization that might get stuck in a poor local minimum. Balancing weights between physics and data terms during training can be challenging and may require a trial-and-error approach.

- *Struggles with Multi-Scale and High-Frequency Solutions*: PINNs tend to approximate smoother, low-frequency solutions better than high-frequency behaviors. This makes solving stiff systems or chaotic regimes particularly difficult unless specialized techniques are employed.

- *Causality and Boundary Conditions*: Standard PINNs often fail to respect the sequential cause-effect nature of time evolution in dynamical systems, especially chaotic systems. Approaches like causality-aware training or domain decomposition are under active investigation to mitigate these problems.

- *Efficiency and Scalability*: Training PINNs can be computationally expensive, particularly for higher-order derivatives in complex systems. This cost can outweigh the benefits for simpler or short-time problems.

- *Reliability and Generalization*: Even if a PINN converges, it may not always reproduce the true physical solution without additional constraints. Non-unique solutions in inverse problems remain a significant challenge.

- *Addressing Chaotic Dynamics*: Capturing chaotic behaviors remains a largely unsolved problem, though strategies like sequential training or adding invariant constraints show promise.

### Future Directions

PINNs are a dynamic and developing field, with ongoing research exploring improved training strategies, adaptability in weighting loss components, and innovative approaches to handle their limitations. Investigating modifications or enhancing capabilities for chaotic systems and long-time horizon problems are particularly active research areas.

### References

1. [What Are Physics-Informed Neural Networks - MathWorks](https://www.mathworks.com/discovery/physics-informed-neural-networks.html#:~:text=What%20Are%20Physics,PINNs)
2. [Resistance Condition of PINNs - AIMS Press](https://www.aimspress.com/article/doi/10.3934/math.2025364#:~:text=resistance%20condition,the%20minimizer%20of%20the%20loss)
3. [Wave Equation in PINNs - MDPI](https://www.mdpi.com/1424-8220/23/5/2792#:~:text=The%20wave%20equation%20has%20been,on%20the%20propagation%20of%20a)
4. [Van der Pol and Chaotic Dynamics - ArXiv](https://arxiv.org/html/2408.11077v1#:~:text=Van%20der%20Pol%2C%20and%20Duffing,and%20can%20even%20predict%20solutions)
5. [Loss Function Challenge in PINNs - ArXiv](https://arxiv.org/abs/2203.07404#:~:text=,loss%20functions%20that%20can%20explicitly)
6. [Ensemble PINNs for Inverse Transport - Bohrium](https://www.bohrium.com/paper-details/ensemble-physics-informed-neural-networks-a-framework-to-improve-inverse-transport-modeling-in-heterogeneous-domains/875705581518717745-3872#:~:text=informed%20neural%20networks%20,set%20of%20patterns%20can%20guide)
