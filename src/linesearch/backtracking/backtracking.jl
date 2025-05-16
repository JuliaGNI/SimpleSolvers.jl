using Printf
@doc raw"""
    const DEFAULT_ARMIJO_α₀

The default starting value for ``\alpha`` used in [`SufficientDecreaseCondition`](@ref) (also see [`BacktrackingState`](@ref) and [`QuadraticState`](@ref)).
Its value is """ * """$(DEFAULT_ARMIJO_α₀)
"""
const DEFAULT_ARMIJO_α₀ = 1.0

"""
    const DEFAULT_ARMIJO_σ₀

Constant used in [`QuadraticState`](@ref). Also see [`DEFAULT_ARMIJO_σ₁`](@ref).

It is meant to *safeguard against stagnation* when performing line searches (see [kelley1995iterative](@cite)).

Its value is $(DEFAULT_ARMIJO_σ₀)
"""
const DEFAULT_ARMIJO_σ₀ = 0.1

"""
    const DEFAULT_ARMIJO_σ₁

Constant used in [`QuadraticState`](@ref). Also see [`DEFAULT_ARMIJO_σ₀`](@ref).
Its value is $(DEFAULT_ARMIJO_σ₁)
"""
const DEFAULT_ARMIJO_σ₁ = 0.5

"""
    const DEFAULT_ARMIJO_p

Constant used in [`BacktrackingState`](@ref).
Its value is $(DEFAULT_ARMIJO_p)
"""
const DEFAULT_ARMIJO_p  = 0.5

@doc raw"""
    const DEFAULT_WOLFE_c₁

A constant ``\epsilon`` on which a finite difference approximation of the derivative of the objective is computed. This is then used in the following stopping criterion:

```math
\frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha\cdot{}f'(\alpha_0).
```

# Extended help


"""
const DEFAULT_WOLFE_c₁  = 1E-4

@doc raw"""
    BacktrackingState <: LinesearchState

Corresponding [`LinesearchState`](@ref) to [`Backtracking`](@ref).

# Keys

The keys are:
- `config::`[`Options`](@ref)
- `α₀`: 
- `ϵ=$(DEFAULT_WOLFE_c₁)`: a default step size on whose basis we compute a finite difference approximation of the derivative of the objective. Also see [`DEFAULT_WOLFE_c₁`](@ref).
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

!!! info
    The algorithm allocates an instance of `SufficientDecreaseCondition` by calling `SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, one(α), obj)`, here we take the *value one* for the search direction ``p``, this is because we already have the search direction encoded into the line search objective.
"""
struct BacktrackingState{OPT <: Options, T <: Number} <: LinesearchState
    config::OPT
    α₀::T
    ϵ::T
    p::T

    function BacktrackingState(::Type{T₁}=Float64; config::Options = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_WOLFE_c₁,
                    p::T = DEFAULT_ARMIJO_p) where {T₁, T}
        @assert p < 1 "The shrinking parameter needs to be less than 1, it is $(p)."
        @assert ϵ < 1 "The search control parameter needs to be less than 1, it is $(ϵ)."
        configT = Options(T₁, config)
        new{typeof(configT), T₁}(configT, T₁(α₀), T₁(ϵ), T₁(p))
    end
end

Base.show(io::IO, ls::BacktrackingState) = print(io, "Backtracking with α₀ = " * string(ls.α₀) * ", ϵ = " * string(ls.ϵ) * "and p = " * string(ls.p) * ".")

LinesearchState(algorithm::Backtracking; T::DataType = Float64, kwargs...) = BacktrackingState(T; kwargs...)

function (ls::BacktrackingState{OT, T})(obj::AbstractUnivariateObjective{T}, α::T = ls.α₀) where {OT, T}
    x₀ = zero(α)
    y₀ = value!(obj, x₀)
    d(α) = derivative!(obj, α)
    d₀ = d(x₀)

    # note that we set pₖ ← 0 here as this is the descent direction for the linesearch objective.
    sdc = SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, one(α), obj)
    cc = CurvatureCondition(T(.9), x₀, d₀, one(α), obj, d; mode=:Standard)
    for _ in 1:ls.config.max_iterations
        if (sdc(α) && cc(α))
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