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
const DEFAULT_WOLFE_c₁ = 1E-4

@doc raw"""
    const DEFAULT_WOLFE_c₂

The constant used in the second Wolfe condition (the [`CurvatureCondition`](@ref)). According to [nocedal2006numerical,kochenderfer2019algorithms](@cite) we should have
```math
c_2 \in (c_1, 1).
```
Furthermore [nocedal2006numerical](@cite) recommend ``c_2 = 0.9`` and [kochenderfer2019algorithms](@cite) write that "it is common to set [``c_2=0.1``] when approximate line search is used with the conjugate gradient method and to 0.9 when used with Newton's method."
We use ``c_2 = 0.9`` as default.
"""
const DEFAULT_WOLFE_c₂ = 0.9

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The keys are:
- `α₀=$(DEFAULT_ARMIJO_α₀)`: the initial step size ``\alpha``. This is decreased iteratively by a factor ``p`` until the Wolfe conditions (the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref)) are satisfied.
- `c₁=$(DEFAULT_WOLFE_c₁)`: a default step size on whose basis we compute a finite difference approximation of the derivative of the problem. Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `c₂=$(DEFAULT_WOLFE_c₂)`: the constant on whose basis the [`CurvatureCondition`](@ref) is tested. We should have ``c_2\in(c_1, 1).`` The closer this constant is to 1, the easier it is to satisfy the [`CurvatureCondition`](@ref).
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
    f(\alpha) < y_0 + c_1 \cdot \alpha \cdot d_0,
```
where ``c_1`` is stored in `ls`.

!!! info
    The algorithm allocates an instance of `SufficientDecreaseCondition` by calling `SufficientDecreaseCondition(ls.c₁, x₀, y₀, d₀, one(α), obj)`, here we take the *value one* for the search direction ``p``, this is because we already have the search direction encoded into the line search problem.

# Extended help

The backtracking algorithm starts by setting ``y_0 \gets f(0)`` and ``d_0 \gets \nabla_0f``.

The algorithm is executed by calling the functor of [`Backtracking`](@ref).

The following is then repeated until the stopping criterion is satisfied or `config.max_iterations` """ * """($(MAX_ITERATIONS) by default) is reached:

```julia
if value(obj, α) ≥ y₀ + ls.c₁ * α * d₀
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
    c₁::T
    c₂::T
    p::T

    function Backtracking{T}(α₀::T, c₁::T, c₂::T, p::T) where {T}
        @assert p < 1 "The shrinking parameter needs to be less than 1, it is $(p)."
        @assert c₁ < 1 "The search control parameter needs to be less than 1, it is $(c₁)."
        new{T}(α₀, c₁, c₂, p)
    end
end

function Backtracking(::Type{T}=Float64;
    α₀=T(DEFAULT_ARMIJO_α₀),
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    p=T(DEFAULT_ARMIJO_p)
) where {T}
    Backtracking{T}(α₀, c₁, c₂, p)
end

Backtracking(::Type{T}, ::SolverMethod) where {T} = Backtracking(T)


# function solve(ls::Linesearch{T,<:Backtracking}, α::T=method(ls).α₀) where {T,LST}
function solve(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    f(α) = value(problem(ls), α, params)
    d(α) = derivative(problem(ls), α, params)

    α₀ = zero(α)
    y₀ = f(α₀)
    d₀ = d(α₀)

    # note that we set pₖ ← 0 here as this is the descent direction for the linesearch problem.
    sdc = SufficientDecreaseCondition(method(ls).c₁, y₀, d₀, f)
    cc = CurvatureCondition(method(ls).c₂, d₀, d; mode=:Standard)

    for i in 1:config(ls).max_iterations
        if (sdc(α) && cc(α))
            break
        else
            α *= method(ls).p
        end
    end

    α
end

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with α₀ = $(ls.α₀) c₁ = $(ls.c₁), c₂ = $(ls.c₂) and p = $(ls.p).")

function Base.convert(::Type{T}, method::Backtracking) where {T}
    T ≠ eltype(method) || return method
    Backtracking{T}(T(method.α₀), T(method.c₁), T(method.c₂), T(method.p))
end

function Base.isapprox(bt₁::Backtracking{T}, bt₂::Backtracking{T}; kwargs...) where {T}
    isapprox(bt₁.α₀, bt₂.α₀; kwargs...) && isapprox(bt₁.c₁, bt₂.c₁; kwargs...) && isapprox(bt₁.c₂, bt₂.c₂; kwargs...) && isapprox(bt₁.p, bt₂.p; kwargs...)
end
