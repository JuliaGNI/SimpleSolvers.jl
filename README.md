# SimpleSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/dev)
[![Build Status](https://github.com/JuliaGNI/SimpleSolvers.jl/workflows/CI/badge.svg)](https://github.com/JuliaGNI/SimpleSolvers.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl)

This package provides simple linear and nonlinear solvers such as LU decomposition and Newton's method. Under a unified interface, it provides low-overhead implementations in pure Julia, applicable to a wide range of data types, and wraps methods from other Julia libraries. Nonlinear solvers can be used with linesearch algorithms. Jacobians can be computed via automatic differentiation, finite differences or manually.
