[![CI](https://github.com/ultimatekarate/volterra-stability/actions/workflows/ci.yml/badge.svg)](https://github.com/ultimatekarate/volterra-stability/actions/workflows/ci.yml)

# volterra-stability

Domain-agnostic Rust library for stability analysis of distributed systems modeled as coupled [Volterra integral equations](https://en.wikipedia.org/wiki/Volterra_integral_equation) of the second kind with exponential decay kernels.

## What it does

Given a system of coupled integrals with exponential decay, this library provides multiple layers of stability certification:

- **Eigenvalue analysis** -- verify all Re(lambda) < 0 for exponential decay
- **Spectral gap analysis** -- spectral gap, eigenvector conditioning, Henrici departure from normality, and stability radius
- **Dyson series** -- transient perturbation response and cascade analysis for sequential threats
- **Contractivity proofs** -- energy dissipation verification under a Lyapunov matrix P
- **Lyapunov exponents** -- maximal exponent via Benettin's method for nonlinear stability

## Architecture

The crate is structured as two layers with strict purity guarantees (no I/O, no async, no filesystem access):

**Dictionary layer** -- inert data structures, no computation:
- `config` -- system configuration (`SystemConfig`, `ChannelConfig`, `OperatingPoint`, `ImpulseRates`)
- `report` -- output types (`StabilityReport`, `SpectralGapReport`, `DysonAnalysisReport`, etc.)

**Laboratory layer** -- pure mathematics:
- `integral` -- O(1) decaying integral evaluation (`DecayingIntegral`, `IntegralBank`)
- `coupling` -- `CouplingModel` trait for domain-specific coupling structure
- `scaler` -- dimensionless pressure signal normalization
- `jacobian` -- Jacobian construction and scenario analysis entry point
- `eigenvalues` -- eigenvalue stability and contractivity checks
- `spectral` -- spectral gap, stability radius (3-stage omega-sweep), decay guarantees
- `pade` -- matrix exponential via Pade(13) scaling-and-squaring (Higham 2005)
- `dyson` -- Dyson series with 16-point Gauss-Legendre quadrature, cascade analysis
- `integrators` -- RK4 and adaptive Dormand-Prince RK4(5) with PI step control
- `nonlinear` -- Benettin's renormalization method for maximal Lyapunov exponent
- `contractivity` -- const-fn Cholesky factorization for compile-time contractivity proofs

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
volterra-stability = { path = "../volterra-stability" }
```

### 1. Define a coupling model

Implement the `CouplingModel` trait to describe how your system's channels interact:

```rust
use volterra_stability::coupling::CouplingModel;
use volterra_stability::config::*;
use nalgebra::DMatrix;

struct MyCouplingModel { /* ... */ }

impl CouplingModel for MyCouplingModel {
    fn config(&self) -> &SystemConfig { /* ... */ }

    fn build_jacobian(&self, rates: &ImpulseRates, op: &OperatingPoint) -> DMatrix<f64> {
        // J[i,j] = df_i/dI_j - lambda_i * delta_{ij}
        // Define how channel j's integral value affects channel i's rate
        todo!()
    }

    fn normalization_scales(&self) -> Vec<f64> { /* ... */ }
    fn lyapunov_matrix(&self) -> Option<DMatrix<f64>> { None }
}
```

### 2. Run stability analysis

```rust
use volterra_stability::jacobian;

let report = jacobian::analyze_scenario("my_scenario", &model, &rates, &operating_point);
assert!(report.is_stable);
```

### 3. Analyze spectral properties

```rust
use volterra_stability::spectral;

let jacobian = model.build_jacobian(&rates, &operating_point);
let spectral_report = spectral::analyze_spectral_gap("my_scenario", &jacobian);
// spectral_report.gamma_1 -- fastest decay rate
// spectral_report.stability_radius -- robustness to perturbations
```

### 4. Simulate threat responses

```rust
use volterra_stability::dyson::{self, ThreatProfile};

let threat = ThreatProfile {
    name: "spike".into(),
    onset: 0.0,
    duration: 1.0,
    forcing: vec![1.0, 0.0, 0.0],
    coupling_delta: None,
};
let response = dyson::impulse_response(&jacobian, &[threat], "spike_test");
```

## Mathematical foundations

The core model is a system of coupled integrals with exponential decay:

```
I_i(t) = integral_0^t exp(-lambda_i * (t - s)) * f_i(I(s), r(s)) ds
```

Stability analysis proceeds through the Jacobian at an operating point:

```
J[i,j] = df_i/dI_j - lambda_i * delta_{ij}
```

**Eigenvalue stability**: all eigenvalues satisfy Re(lambda) < 0.

**Spectral gap**: gamma_1 = min |Re(lambda)| measures the slowest decay rate; gamma_2 - gamma_1 measures transient suppression.

**Stability radius**: r(J) = min_omega sigma_min(i*omega*I - J) gives the smallest perturbation that can destabilize the system.

**Contractivity**: for Lyapunov matrix P, if Q = PJ + J^T P is negative definite, then V(x) = x^T P x is a strict Lyapunov function guaranteeing global exponential stability.

**Lyapunov exponents**: Benettin's method integrates perturbation dynamics with periodic renormalization. mu_1 < 0 certifies nonlinear stability.

## Dependencies

- [`nalgebra`](https://crates.io/crates/nalgebra) -- linear algebra (eigenvalues, SVD, LU)
- [`serde`](https://crates.io/crates/serde) -- serialization
- [`thiserror`](https://crates.io/crates/thiserror) -- error types

## License

MIT.
