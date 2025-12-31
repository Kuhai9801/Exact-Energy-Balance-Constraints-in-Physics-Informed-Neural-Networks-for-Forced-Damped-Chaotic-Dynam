# Methodology


---

## 1. Problem Formulation

This work addresses physics-informed trajectory reconstruction from sparse, noisy, and partially observed measurements of a chaotic forced dynamical system.

### 1.1 State and Observations

The latent continuous-time state is denoted:

$$
\mathbf{y}(t) = \begin{bmatrix} x(t) \\ v(t) \end{bmatrix}, \quad t \in [0, T_{\text{end}}]
$$

The observation set comprises scalar position measurements corrupted by additive Gaussian noise:

$$
\mathcal{D} = \lbrace (t_i, x_{\text{obs}}(t_i)) \rbrace_{i=1}^{N_{\text{obs}}}
$$

$$
x_{\text{obs}}(t_i) = x(t_i) + \eta_i, \quad \eta_i \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)
$$

### 1.2 Observation Protocol

- **Temporal grid spacing:** $\Delta t\_{\text{obs}} = 0.05$
- **Assimilation window:** $T\_{\text{end}} = 100 \, T\_f$ where $T\_f = 2\pi/\omega$
- **Number of gaps:** $G = 10$ non-overlapping contiguous gaps
- **Minimum gap separation:** equal to gap duration $q$

### 1.3 Gap Regimes

Two gap regimes are investigated:

- **Transition regime:** $q = 1 \, T\_f$
- **Deep shadowing regime:** $q = 4 \, T\_f$

The objective is to infer a continuous trajectory that is simultaneously consistent with the noisy measurements and the governing dynamics.

> **Note on identifiability:** For $q \gg \tau\_\lambda$, orbit identity is not identifiable from $x(t)$ alone; we therefore treat reconstruction as regularized smoothing and do not claim recovery of the realized gap itinerary.

---

## 2. Forced-System Invariance

For the periodically forced Duffing system, the invariant object is defined in an extended (autonomized) phase space. Introducing the forcing phase $\phi(t) = \omega t \pmod{2\pi}$ yields the extended state $\mathbf{z} = [x, v, \phi]^\top$.

To avoid discontinuities at $\phi = 0$, we employ the continuous embedding:

$$
\mathbf{z}(t) = \big[ x(t), v(t), \cos(\omega t), \sin(\omega t) \big]^\top
$$

---

## 3. Lyapunov Regime Verification

Chaoticity defines a predictability scale via:

- **Maximal Lyapunov exponent:** $\lambda\_{\max} > 0$
- **Lyapunov time:** $\tau\_\lambda \approx 1/\lambda\_{\max}$

### 3.1 Stroboscopic Rosenstein Algorithm

The exponent is estimated from a model-only reference simulation using the stroboscopic map $P: \mathbf{y}(t) \to \mathbf{y}(t + T\_f)$.

Let $\mathbf{y}\_n = \mathbf{y}(t\_0 + n T\_f)$ for $n = 0, \ldots, N\_{\text{strobe}} - 1$ with $N\_{\text{strobe}} = 2000$. We set $t\_0$ to the start of the post-burn-in reference window (i.e., $t\_0 = 0$ in the saved reference trajectory). The stroboscopic series $\lbrace \mathbf{y}\_n \rbrace$ is extracted from the same long model-only reference trajectory used for characteristic scaling (Section 6.5). Let $k\_{\max} = 8$ and define $\mathcal{I} := \lbrace 0, 1, \ldots, N\_{\text{strobe}} - 1 - k\_{\max} \rbrace$ so that $\mathbf{y}\_{n+k}$ is well-defined for all $n \in \mathcal{I}$ and $k \leq k\_{\max}$.

**Algorithm specifications:**

- Euclidean distance metric in $\mathbb{R}^2$
- Theiler window: $w = 1$ iterate (verified for $w \in \lbrace 1, 2 \rbrace$)
- Linear fit region: $k \in \lbrace 1, \ldots, 8 \rbrace$
- Linearity criterion: $R^2 \geq 0.98$

For each index $n$, select nearest neighbor $n'$ (excluding self-match and respecting Theiler window) minimizing distance subject to $|n - n'| \geq w$. Compute mean log-separation growth:

$$
S(k) = \frac{1}{|\mathcal{I}|} \sum_{n \in \mathcal{I}} \log \lVert \mathbf{y}_{n+k} - \mathbf{y}_{n'+k} \rVert_2
$$

Fit $S(k) \approx S(0) + \lambda\_{\text{map}} \cdot k$ over $k = 1, \ldots, 8$.

### 3.2 Continuous-Time Conversion

$$
\lambda_{\max} \approx \frac{\lambda_{\text{map}}}{T_f}, \quad \tau_\lambda \approx \frac{1}{\lambda_{\max}}
$$

**Measured values:**

- $\lambda\_{\max} \approx 0.16 \, \text{s}^{-1}$
- $\tau\_\lambda \approx 6.3 \, \text{s}$
- Transition: $q = T\_f \approx 0.83 \, \tau\_\lambda$
- Deep shadowing: $q = 4 T\_f \approx 3.3 \, \tau\_\lambda$

---

## 4. Governing Dynamics

The forced-damped Duffing oscillator:

$$
\dot{x} = v
$$

$$
\dot{v} = -\delta v - \alpha x - \beta x^3 + \gamma \cos(\omega t)
$$

### 4.1 Parameters (Standard Chaotic Regime)

- $\alpha = -1$ (linear stiffness)
- $\beta = 1$ (cubic nonlinearity)
- $\delta = 0.3$ (damping coefficient)
- $\gamma = 0.5$ (forcing amplitude)
- $\omega = 1.2$ (forcing frequency)
- $T\_f = 2\pi/\omega \approx 5.236$ (forcing period)

---

## 5. Energetic Preconditioning

### 5.1 Mechanical Energy

$$
H(x, v) = \frac{1}{2} v^2 + \frac{1}{2} \alpha x^2 + \frac{1}{4} \beta x^4
$$

### 5.2 Energy Balance

Along exact trajectories:

$$
\frac{dH}{dt} = \gamma v \cos(\omega t) - \delta v^2
$$

### 5.3 Power-Balance Residual

$$
\mathcal{R}\_{\text{power}}(t) = \frac{dH}{dt} - \big( \gamma v \cos(\omega t) - \delta v^2 \big)
$$

Denoting the vector field $\mathbf{F}(\mathbf{y}, t) = [v, -\delta v - \alpha x - \beta x^3 + \gamma \cos(\omega t)]^\top$ and energy gradient $\nabla H(\mathbf{y}) = [\alpha x + \beta x^3, v]^\top$:

$$
\mathcal{R}_{\text{power}}(t) \equiv \nabla H(\mathbf{y}) \cdot \big( \dot{\mathbf{y}} - \mathbf{F}(\mathbf{y}, t) \big)
$$

Minimizing $|\mathcal{R}\_{\text{power}}|^2$ induces a rank-1, state-dependent quadratic penalty on the vector-field error, concentrating optimization pressure where $\lVert \nabla H \rVert$ is large.

---

## 6. Physics-Informed Neural Network

### 6.1 Architecture

- **Layers:** 4 hidden layers of width 64
- **Activation:** $\tanh$
- **Initialization:** Xavier-uniform weights, zero biases
- **Precision:** float64

### 6.2 Time Normalization

$$
\tilde{t} = \frac{2t}{T_{\text{end}}} - 1 \in [-1, 1]
$$

Derivatives in $r\_1, r\_2$ are taken with respect to physical time $t$. With $\tilde{t} = 2t/T\_{\text{end}} - 1$, we compute $\frac{d}{dt} = \frac{2}{T\_{\text{end}}} \frac{d}{d\tilde{t}}$ via the chain rule (and analogously for higher components).

### 6.3 Fourier Features

Input augmented with bounded trainable Fourier features:

$$
\varphi(t) = \big[ \sin(\omega\_k t), \cos(\omega\_k t) \big]\_{k=1}^{K}, \quad K = 16
$$

- Frequencies: $\omega\_k = \omega\_{\max} \cdot \sigma(\xi\_k)$ with $\omega\_{\max} = 25$
- Initialization: log-uniform over $[0.5\omega, 10\omega]$
- **Ablation:** Fixed harmonics $\omega\_k \in \lbrace \omega, 3\omega, \ldots, (2K-1)\omega \rbrace$

### 6.4 Residual Definitions

$$
r_1(t) = \dot{x}_\theta(t) - v_\theta(t)
$$

$$
r_2(t) = \dot{v}_\theta(t) + \delta v_\theta(t) + \alpha x_\theta(t) + \beta x_\theta^3(t) - \gamma \cos(\omega t)
$$

### 6.5 Characteristic Scaling

Computed from model-only attractor simulation (fixed across all experiments). Reference run length $T\_{\text{ref}} = 500 \, T\_f$ after burn-in, sampled at $dt = 0.01$.

$$
s_{r1} = \text{RMS}(|v_{\text{ref}}(t)|), \quad s_{r2} = \text{RMS}(|\dot{v}_{\text{ref}}(t)|)
$$

$$
s\_{dH} = \text{RMS}\big( |\gamma v\_{\text{ref}}(t) \cos(\omega t) - \delta v\_{\text{ref}}^2(t)| \big)
$$

Here $\dot{v}\_{\text{ref}}(t)$ is evaluated analytically from the Duffing vector field (no numerical differentiation).

---

## 7. Loss Function

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_p \mathcal{L}_{\text{phys}} + \lambda_e \mathcal{L}_{\text{power}} + \lambda_{ic} \mathcal{L}_{ic}
$$

**Weights:** $\lambda\_p = 1$, $\lambda\_e = 0.2$, $\lambda\_{ic} = 0.1$

### 7.1 Data Term (Gaussian Likelihood)

$$
\mathcal{L}\_{\text{data}} = \frac{1}{N\_{\text{obs}}} \sum\_{i=1}^{N\_{\text{obs}}} \left( \frac{x\_\theta(t\_i) - x\_{\text{obs}}(t\_i)}{\sigma\_{\text{noise}}} \right)^2
$$

### 7.2 Physics Residuals

$$
\mathcal{L}_{\text{phys}} = \frac{1}{N_{\text{col}}} \sum_{c=1}^{N_{\text{col}}} \left[ \left( \frac{r_1(t_c)}{s_{r1}} \right)^2 + \left( \frac{r_2(t_c)}{s_{r2}} \right)^2 \right]
$$

### 7.3 Power-Balance Preconditioning

$$
\mathcal{L}_{\text{power}} = \frac{1}{N_{\text{col}}} \sum_{c=1}^{N_{\text{col}}} \left( \frac{\mathcal{R}_{\text{power}}(t_c)}{s_{dH}} \right)^2
$$

### 7.4 Soft Initial Anchor

$$
\mathcal{L}_{ic} = \left( \frac{x_\theta(t_{\min}) - x_{\text{obs}}(t_{\min})}{\sigma_{\text{noise}}} \right)^2
$$

Here $t\_{\min} := \min\_i t\_i$ is the earliest observation time (equal to $0$ by the gap constraint in Section 10.2).

---

## 8. Optimization

### 8.1 Collocation Sampling

- $N\_{\text{col}} = 8192$ points per iteration
- **Mixture:** 70% gap interiors, 30% observed intervals

Here "gap interiors" denote the union of open missing intervals $(t\_L, t\_R)$ (excluding gap endpoints), and collocation times are sampled uniformly in continuous time within each interval; "observed intervals" denote the complement in $[0, T\_{\text{end}}]$.

### 8.2 Two-Stage Optimization

1. **Adam:** $\text{lr} = 10^{-3}$, $\beta\_1 = 0.9$, $\beta\_2 = 0.999$, 50,000 iterations

The data term $\mathcal{L}\_{\text{data}}$ is evaluated on all $N\_{\text{obs}}$ observations each iteration (no minibatching over observations).

2. **L-BFGS-B:** max 10,000 iterations, history size 50

### 8.3 Determinism Constraint

L-BFGS-B uses a **frozen** collocation set of size $N\_{\text{col}} = 65536$ to ensure deterministic objective (Adam uses $N\_{\text{col}} = 8192$ per iteration; L-BFGS-B uses a frozen set of size 65536). The frozen collocation set is drawn once from the same 70% gap / 30% observed mixture and held fixed throughout L-BFGS-B.

---

## 9. Baselines

### 9.1 Locally Periodic Gaussian Process

$$
k(t, t') = \sigma_k^2 \exp\left( -\frac{2 \sin^2(\pi(t-t')/T_f)}{\ell_{\text{per}}^2} \right) \exp\left( -\frac{(t-t')^2}{2 \ell_{\text{rbf}}^2} \right)
$$

- Period fixed to $T\_f$
- $M = 512$ inducing points
- Adam optimizer: $\text{lr} = 0.05$, 3,000 steps

Kernel hyperparameters $(\sigma\_k, \ell\_{\text{per}}, \ell\_{\text{rbf}})$ are fit by maximizing the (sparse variational) marginal likelihood with observation noise variance fixed to $\sigma\_{\text{noise}}^2$. We report $v\_{\text{GP}}(t) = \frac{d}{dt}\mu(t)$, the analytic derivative of the posterior mean.

### 9.2 Strong-Constraint IVP Shooting

$$
\min_{x_0, v_0} \sum_{i=1}^{N_{\text{obs}}} \left( \frac{x(t_i; x_0, v_0) - x_{\text{obs}}(t_i)}{\sigma_{\text{noise}}} \right)^2
$$

- **Integrator:** DOP853 with $\text{rtol} = 10^{-10}$, $\text{atol} = 10^{-12}$
- **Optimizer:** L-BFGS-B with bounds $x\_0 \in [-3, 3]$, $v\_0 \in [-3, 3]$
- **Initialization:** $x\_0 = x\_{\text{obs}}(0)$, $v\_0 = 0$

### 9.3 Weak-Constraint 4D-Var

Time is discretized on a uniform grid $t\_k = k \Delta t$ with step $\Delta t = 0.01$. The mapping $k(i)$ denotes the grid index corresponding to observation time $t\_i$. Since $t\_i$ lie on the $\Delta t = 0.01$ grid (with $\Delta t\_{\text{obs}} = 0.05$ a multiple of $\Delta t$), we take $k(i) = t\_i / \Delta t$ (integer-valued). Decision variables are $\lbrace \mathbf{y}\_k \rbrace\_{k=0}^{N\_t}$ with $\mathbf{y}\_k = [x\_k, v\_k]^\top$. Initialization uses cubic-spline interpolation of $x$ through observed points and $v$ from finite differences on observed segments (set to 0 inside gaps before optimization).

$$
\mathcal{J}(\lbrace \mathbf{y}_k \rbrace) = \sum_{i \in \mathcal{I}_{\text{obs}}} \left( \frac{x_{k(i)} - x_{\text{obs}}(t_i)}{\sigma_{\text{noise}}} \right)^2 + \lambda_m \sum_{k=0}^{N_t-1} \lVert \mathbf{W}_{\text{step}}^{-1} \big( \mathbf{y}_{k+1} - \Phi_{\Delta t}(\mathbf{y}_k) \big) \rVert_2^2
$$

$$
\mathbf{W}_{\text{step}} = \text{diag}(\Delta t \cdot s_{r1}, \Delta t \cdot s_{r2}), \quad \lambda_m = 1
$$

- **Flow map:** RK4 one-step integrator $\Phi\_{\Delta t}$
- **Optimizer:** L-BFGS-B with max 20,000 iterations

---

## 10. Experimental Protocol

### 10.1 Data Generation

- **Integrator:** DOP853 with $\text{rtol} = \text{atol} = 10^{-12}$
- **Initial conditions:** uniform on $[-2, 2] \times [-2, 2]$
- **Burn-in:** $T\_{\text{burn}} = 200 \, T\_f$
- **Window:** $T\_{\text{end}} = 100 \, T\_f$
- **Resample step:** $dt = 0.01$
- **Observation step:** $\Delta t\_{\text{obs}} = 0.05$
- **Noise levels:** $\sigma\_{\text{noise}} \in \lbrace 0.02, 0.05, 0.10 \rbrace$
- **Random seeds:** $S = 10$. The 10 random seeds control noise realization, gap placement, network initialization, and collocation RNG.

### 10.2 Gap Configuration

- $G = 10$ gaps
- $q \in \lbrace T\_f, 4T\_f \rbrace$
- **Minimum separation:** $q$
- **No overlap**
- Gap endpoints are aligned to the observation grid $\Delta t\_{\text{obs}} = 0.05$
- Gaps are constrained not to include $t = 0$, so the earliest observation time satisfies $t\_{\min} = 0$ in all experiments

---

## 11. Evaluation Metrics

### 11.1 Gap-RMS

RMSE of $x\_\theta(t)$ against ground truth on gap interiors with 5% boundary trimming. All gap-interior metrics are evaluated on the uniform $dt = 0.01$ grid (with 5% trimming for Gap-RMS).

### 11.2 Stroboscopic Lyapunov Exponent

Rosenstein estimate with Theiler $w = 1$ and fit range $k = 1, \ldots, 8$:

$$
\hat{\lambda}_{\max} = \hat{\lambda}_{\text{map}} / T_f
$$

### 11.3 Extended-Attractor MMD

Maximum mean discrepancy (Gaussian kernel, median heuristic bandwidth) between reconstructed gap states and phase-matched reference:

- **Reference set:** $N\_{\text{ref}} = 50000$ samples from $T\_{\text{ref}} = 500 \, T\_f$ post-burn-in trajectory
- **Predicted samples:** $N\_{\text{pred}} = 5000$ gap-interior points per run (subsampled to avoid $O(n^2)$ cost)
- **Phase bins:** 32 uniform bins of $\phi = \omega t \pmod{2\pi}$
- **Bootstrap:** $B = 200$ replicates, block length $T\_f$

For each predicted sample, we draw a reference sample uniformly from the same phase bin (with replacement) to form the phase-matched reference set.

Before computing MMD, all coordinates of $[x, v, \cos(\omega t), \sin(\omega t)]$ are standardized to zero mean and unit variance using the reference-set statistics.

### 11.4 Flow-Consistency Defect

For each gap midpoint $t\_c$, integrate the true dynamics from $\mathbf{y}\_\theta(t\_c)$ forward by $\Delta = \tau\_\lambda$:

$$
E\_{\text{flow}}(t\_c; \Delta) = \lVert \Phi\_\Delta(\mathbf{y}\_\theta(t\_c)) - \mathbf{y}\_\theta(t\_c + \Delta) \rVert\_2
$$

- **Integrator:** DOP853 with $\text{rtol} = 10^{-10}$, $\text{atol} = 10^{-12}$
- **Horizon:** $\Delta = \tau\_\lambda$

### 11.5 Physics Compliance

RMS of normalized residuals $r\_1/s\_{r1}$ and $r\_2/s\_{r2}$ on gap interiors. All gap-interior metrics are evaluated on the uniform $dt = 0.01$ grid.

---

## 12. Ablations

- **Energetic preconditioning:** Full objective vs. $\lambda\_e = 0$
- **Fourier embedding:** Trainable vs. fixed harmonics $\lbrace \omega, 3\omega, \ldots \rbrace$

---

## 13. Reproducibility

All hyperparameters are fixed across seeds. No tuning across noise levels.
