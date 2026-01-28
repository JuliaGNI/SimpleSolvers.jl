using Printf
@doc raw"""
    const DEFAULT_ARMIJO_α₀

The default starting value for ``\alpha`` used in [`SufficientDecreaseCondition`](@ref) (also see [`Backtracking`](@ref) and [`Quadratic`](@ref)).
Its value is """ * """$(DEFAULT_ARMIJO_α₀)
"""
const DEFAULT_ARMIJO_α₀ = 1.0

"""
    const DEFAULT_ARMIJO_σ₀

Constant used in [`Quadratic`](@ref). Also see [`DEFAULT_ARMIJO_σ₁`](@ref).

It is meant to *safeguard against stagnation* when performing line searches (see [kelley1995iterative](@cite)).

Its value is $(DEFAULT_ARMIJO_σ₀)
"""
const DEFAULT_ARMIJO_σ₀ = 0.1

"""
    const DEFAULT_ARMIJO_σ₁

Constant used in [`Quadratic`](@ref). Also see [`DEFAULT_ARMIJO_σ₀`](@ref).
Its value is $(DEFAULT_ARMIJO_σ₁)
"""
const DEFAULT_ARMIJO_σ₁ = 0.5

"""
    const DEFAULT_ARMIJO_p

Constant used in [`Backtracking`](@ref).
Its value is $(DEFAULT_ARMIJO_p)
"""
const DEFAULT_ARMIJO_p = 0.5

@doc raw"""
    const DEFAULT_WOLFE_c₁

A constant ``\epsilon`` on which a finite difference approximation of the derivative of the problem is computed. This is then used in the following stopping criterion:

```math
\frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha\cdot{}f'(\alpha_0).
```

# Extended help


"""
const DEFAULT_WOLFE_c₁  = 1E-4

@doc raw"""
    const DEFAULT_WOLFE_c₂

The constant used in the second Wolfe condition (the [`CurvatureCondition`](@ref)). According to [nocedal2006numerical,kochenderfer2019algorithms](@cite) we should have
```math
c_2 \in (c_1, 1).
```
Furthermore [nocedal2006numerical](@cite) recommend ``c_2 = 0.9`` and [kochenderfer2019algorithms](@cite) write that "it is common to set [``c_2=0.1``] when approximate line search is used with the conjugate gradient method and to 0.9 when used with Newton's method."
We also use ``c_2 =``$(DEFAULT_WOLFE_c₂) here.
"""
const DEFAULT_WOLFE_c₂ = 0.9

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The keys are:
- `α₀`:
- `ϵ=$(DEFAULT_WOLFE_c₁)`: a default step size on whose basis we compute a finite difference approximation of the derivative of the problem. Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `p=$(DEFAULT_ARMIJO_p)`: a parameter with which ``\alpha`` is decreased in every step until the stopping criterion is satisfied.

# Functor

The functor is used the following way:

```julia
ls(obj, α = ls.α₀)
```

# Implementation

The algorithm starts by setting:

```math
x_0 \gets 0,
y_0 \gets f(x_0),
d_0 \gets f'(x_0),
\alpha \gets \alpha_0,
```
where ``f`` is the *univariate optimizer problem* (of type [`LinesearchProblem`](@ref)) and ``\alpha_0`` is stored in `ls`. It then repeatedly does ``\alpha \gets \alpha\cdot{}p`` until either (i) the maximum number of iterations is reached (the `max_iterations` keyword in [`Options`](@ref)) or (ii) the following holds:
```math
    f(\alpha) < y_0 + \epsilon \cdot \alpha \cdot d_0,
```
where ``\epsilon`` is stored in `ls`.

!!! info
    The algorithm allocates an instance of `SufficientDecreaseCondition` by calling `SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, one(α), obj)`, here we take the *value one* for the search direction ``p``, this is because we already have the search direction encoded into the line search problem.

# Extended help

The backtracking algorithm starts by setting ``y_0 \gets f(0)`` and ``d_0 \gets \nabla_0f``.

The algorithm is executed by calling the functor of [`Backtracking`](@ref).

The following is then repeated until the stopping criterion is satisfied or `config.max_iterations` """ * """($(MAX_ITERATIONS) by default) is reached:

```julia
if value(obj, α) ≥ y₀ + ls.ϵ * α * d₀
    α *= ls.p
else
    break
end
```
The stopping criterion as an equation can be written as:

```math
f(\alpha) < y_0 + \epsilon \alpha \nabla_0f = y_0 + \epsilon (\alpha - 0)\nabla_0f.
```
Note that if the stopping criterion is not reached, ``\alpha`` is multiplied with ``p`` and the process continues.

[Sometimes](https://en.wikipedia.org/wiki/Backtracking_line_search) the parameters ``p`` and ``\epsilon`` have different names such as ``\tau`` and ``c``.
"""
struct Backtracking{T} <: LinesearchMethod{T}
    α₀::T
    ϵ::T
    c₂::T
    p::T

    function Backtracking(::Type{T₁}=Float64;
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_WOLFE_c₁,
                    c₂::T = DEFAULT_WOLFE_c₂,
                    p::T = DEFAULT_ARMIJO_p) where {T₁, T}
        @assert p < 1 "The shrinking parameter needs to be less than 1, it is $(p)."
        @assert ϵ < 1 "The search control parameter needs to be less than 1, it is $(ϵ)."
        new{T₁}(T₁(α₀), T₁(ϵ), T(c₂), T₁(p))
    end
end

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with α₀ = " * string(ls.α₀) * ", ϵ = " * string(ls.ϵ) * " and p = " * string(ls.p) * ".")

function solve(obj::LinesearchProblem{T}, ls::Linesearch{T, LST}, α::T=ls.algorithm.α₀) where {T, LST <: Backtracking}
    x₀ = zero(α)
    y₀ = value(obj, x₀)
    d(α) = derivative(obj, α)
    d₀ = d(x₀)

    # note that we set pₖ ← 0 here as this is the descent direction for the linesearch problem.
    sdc = SufficientDecreaseCondition(ls.algorithm.ϵ, x₀, y₀, d₀, one(α), obj)
    cc = CurvatureCondition(T(ls.algorithm.c₂), x₀, d₀, one(α), obj, d; mode=:Standard)
    for _ in 1:ls.config.max_iterations
        if (sdc(α) && cc(α))
            break
        else
            α *= ls.algorithm.p
        end
    end

    α
end

function Base.convert(::Type{T}, algorithm::Backtracking) where {T}
    T ≠ eltype(algorithm) || return algorithm
    Backtracking(T; α₀=T(algorithm.α₀), ϵ=T(algorithm.ϵ), c₂=T(algorithm.c₂), p=T(algorithm.p))
end