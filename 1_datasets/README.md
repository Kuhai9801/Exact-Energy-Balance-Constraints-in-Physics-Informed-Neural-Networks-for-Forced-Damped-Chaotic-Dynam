# Datasets

This repository studies reconstruction of chaotic dynamics in forced–damped nonlinear systems from sparse, noisy, and partially observed measurements. All datasets in this folder are generated to support controlled evaluation of physics-informed neural networks (PINNs) and baselines on the forced–damped Duffing oscillator over a fixed horizon $t \in [0, T_{\text{end}}]$.

## Problem Setting and Intended Use

The learning task is **nonlinear in-domain reconstruction (imputation)**: given noisy displacement observations with substantial temporal dropouts, the objective is to recover a continuous latent trajectory within unobserved gaps by enforcing consistency with the governing physics. Reconstruction is evaluated under two regimes determined by the gap duration relative to the Lyapunov time $\tau_{\lambda} \approx 1 / \lambda_{\max}$.

- In the interpolation regime ( $q \lesssim \tau_{\lambda}$ ), pointwise recovery is meaningful and performance is quantified by RMSE computed strictly on timestamps within reconstruction gaps.
- In the shadowing regime ( $q \gg \tau_{\lambda}$ ), unique phase-aligned recovery is ill-posed and performance emphasizes attractor-consistent shadowing and distributional agreement (e.g., Poincaré-section discrepancies) rather than pointwise error.

Accordingly, this folder contains datasets enabling gap lengths $q \in \{1, 2, 4\}T$, where $T = 2\pi / \omega$ is the forcing period.

## Governing Dynamics Underlying All Data

All synthetic trajectories are derived from the forced–damped Duffing oscillator

$$
\ddot{x}(t) + \delta \dot{x}(t) + \alpha x(t) + \beta x(t)^3 = \gamma \cos(\omega t),
$$

represented in first-order form with $v(t) = \dot{x}(t)$:

$$
\begin{aligned}
\dot{x}(t) &= v(t), \\
\dot{v}(t) &= -\delta v(t) - \alpha x(t) - \beta x(t)^3 + \gamma \cos(\omega t).
\end{aligned}
$$

The parameter tuple $(\alpha, \beta, \delta, \gamma, \omega)$ is fixed to a chaotic parameter regime (reported in the main manuscript/configuration for reproducibility).

## Data Generation Pipeline

### Reference Trajectories (Ground Truth)

A reference trajectory is obtained by numerical integration of the Duffing system using a high-accuracy adaptive Runge–Kutta method with absolute and relative tolerances $10^{-9}$. An initial transient prefix is discarded so that the retained segment lies on the attractor. These reference trajectories provide the ground truth $(x_{\text{true}}(t), v_{\text{true}}(t))$ used for evaluation and for constructing noisy observations.

### Noisy Observation Data

Displacement is sampled at a set of observation timestamps

$$
\mathcal{T}_{\text{all}} = \{t_j\}_{j=1}^{N} \subset [0, T_{\text{end}}],
$$

and corrupted with additive Gaussian noise:

$$
x_{\text{obs}}(t_j) = x_{\text{true}}(t_j) + \epsilon_j, \quad \epsilon_j \sim \mathcal{N}(0, \sigma_x^2).
$$

Only displacement is observed; velocity remains latent and is recovered by the models.

### Dropout Masks and Reconstruction Gaps

Sensor dropout is emulated via a binary mask $m_j \in \{0, 1\}$, producing contiguous reconstruction gaps (intervals where $m_j = 0$). The dataset records the gap structure explicitly so that gap-only metrics can be computed without ambiguity. Each dataset instance stores $\{t_j\}$, $x_{\text{obs}}(t_j)$, masks $m_j$, and (for evaluation) the corresponding $(x_{\text{true}}, v_{\text{true}})$. During PINN training, the data-misfit term uses only observed timestamps ( $m_j = 1$ ), while physics constraints are enforced during training via collocation points sampled uniformly from $[0, T_{\text{end}}]$.

### Fixed Validation Mask

To mitigate selection bias and ensure comparability across model variants, a fixed validation mask is applied. A fixed subset (10%) of timestamps with $m_j = 1$ is held out prior to training; this holdout set is generated once and reused identically across all model variants and random seeds. Early stopping and hyperparameter selection use RMSE on held-out observed points only. Reconstruction gaps and physics residuals are excluded from model selection.

## Dataset Contents and Structure

The datasets in this folder are organized to support three roles:

1. Ground truth reference dynamics include the time grid and true states $(x_{\text{true}}, v_{\text{true}})$, along with the governing parameters and simulation metadata (solver tolerances, transient discard interval).
2. Observation-level data include sampled timestamps and noisy displacement $x_{\text{obs}}$, along with the noise scale $\sigma_x$.
3. Masking and split metadata include the dropout mask $m_j$, explicit gap intervals, and validation holdout indices/mask, enabling consistent training/validation separation and gap-restricted evaluation.

