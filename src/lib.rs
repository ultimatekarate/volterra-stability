//! Domain-agnostic Volterra integral stability analysis.
//!
//! This crate extracts the mathematical core of pressure-integral stability
//! analysis from Phalanx into a standalone library. Any distributed system
//! modeled as coupled Volterra integrals of the second kind with exponential
//! decay kernels can use this machinery to:
//!
//! - Compute eigenvalue stability (all Re(λ) < 0)
//! - Analyze spectral gap, stability radius, and eigenvector conditioning
//! - Run Dyson series transient threat analysis
//! - Prove contractivity under a Lyapunov matrix P
//! - Compute maximal Lyapunov exponents via Benettin's method
//!
//! # Architecture
//!
//! The crate follows the Linguistic Code Model:
//! - **Dictionary layer** (`config`, `report`): inert nouns, no computation
//! - **Laboratory layer** (everything else): pure math, no IO

// Dictionary layer — inert nouns
pub mod config;
pub mod report;

// Laboratory layer — pure math
pub mod integral;
pub mod scaler;
pub mod coupling;
pub mod jacobian;
pub mod eigenvalues;
pub mod spectral;
pub mod pade;
pub mod dyson;
pub mod integrators;
pub mod nonlinear;
pub mod contractivity;
