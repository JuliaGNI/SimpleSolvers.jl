using Printf

const DEFAULT_ARMIJO_α₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_p  = 0.5

@doc raw"""
    const DEFAULT_WOLFE_ϵ

A constant ``\epsilon`` on which a finite difference approximation of the derivative of the objective is computed. This is then used in the following stopping criterion:

```math
\frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha\cdot{}f'(\alpha_0).
```
"""
const DEFAULT_WOLFE_ϵ  = 1E-4

@doc raw"""
    BacktrackingState <: LinesearchState

Corresponding [`LinesearchState`](@ref) to [`Backtracking`](@ref).

# Keys

The keys are:
- `config::`[`Options`](@ref)
- `α₀`: 
- `ϵ=$(DEFAULT_WOLFE_ϵ)`: a default step size on whose basis we compute a finite difference approximation of the derivative of the objective. Also see [`DEFAULT_WOLFE_ϵ`](@ref).
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
where ``f`` is the *univariate objective* (of type [`AbstractUnivariateObjective`](@ref)) and ``\alpha_0`` is stored in `ls`. It then repeatedly does ``\alpha \gets \alpha\cdot{}p`` until either (i) the maximum number of iterations is reached (the `max_iterations` keyword in [`Options`](@ref)) or (ii) the following holds:
```math
    f(\alpha) < y_0 + \epsilon \cdot \alpha \cdot d_0,
```
where ``\epsilon`` is stored in `ls`.
"""
struct BacktrackingState{OPT <: Options, T <: Number} <: LinesearchState
    config::OPT
    α₀::T
    ϵ::T
    p::T

    function BacktrackingState(; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_WOLFE_ϵ,
                    p::T = DEFAULT_ARMIJO_p) where {T}
        @assert p < 1 "The shrinking parameter needs to be less than 1, it is $(p)."
        @assert ϵ < 1 "The search control parameter needs to be less than 1, it is $(ϵ)."
        new{typeof(config), T}(config, α₀, ϵ, p)
    end
end

Base.show(io::IO, ls::BacktrackingState) = print(io, "Backtracking with α₀ = " * string(ls.α₀) * ", ϵ = " * string(ls.ϵ) * "and p = " * string(ls.p) * ".")

LinesearchState(algorithm::Backtracking; kwargs...) = BacktrackingState(; kwargs...)

function (ls::BacktrackingState)(obj::AbstractUnivariateObjective, α = ls.α₀)
    local x₀ = zero(α)
    local y₀ = value!(obj, x₀)
    local d₀ = derivative!(obj, x₀)

    sdc = SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, d₀, obj)
    for _ in 1:ls.config.max_iterations
        if sdc(α)
            break
        else
            print(α)
            α *= ls.p
        end
    end

    α
end

backtracking(o::AbstractUnivariateObjective, args...; kwargs...) = BacktrackingState(; kwargs...)(o, args...)
backtracking(f::Callable, g::Callable, args...; kwargs...) = BacktrackingState(; kwargs...)(TemporaryUnivariateObjective(f, g), args...)