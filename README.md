# SimpleSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/latest)
[![PkgEval Status](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSolvers.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSolvers.html)
[![Build Status](https://github.com/JuliaGNI/SimpleSolvers.jl/workflows/CI/badge.svg)](https://github.com/JuliaGNI/SimpleSolvers.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4317189.svg)](https://doi.org/10.5281/zenodo.4317189)

This package provides simple linear and nonlinear solvers such as LU decomposition and Newton's method. Under a unified interface, it provides low-overhead implementations in pure Julia, applicable to a wide range of data types, and wraps methods from other Julia libraries. Nonlinear solvers can be used with linesearch algorithms. Jacobians can be computed via automatic differentiation, finite differences or manually.

## References

If you use SimpleSolvers.jl in your work, please consider citing it by

```
@misc{Kraus:2020:SimpleSolvers,
  title={SimpleSolvers.jl: Simple linear and nonlinear solvers in Julia},
  author={Kraus, Michael},
  year={2020},
  howpublished={\url{https://github.com/JuliaGNI/SimpleSolvers.jl}},
  doi={10.5281/zenodo.4317189}
}
```
