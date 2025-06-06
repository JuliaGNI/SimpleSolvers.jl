"""
    DEFAULT_BIERLAIRE_ε

A constant that determines the *precision* in [`BierlaireQuadraticState`](@ref). The constant recommended in [bierlaire2015optimization](@cite) is `1E-3`.

Note that this constant may also depend on whether we deal with optimizers or solvers.

!!! warn
    We have deactivated the use of this constant for the moment and are only using `eps(T)` in `BierlaireQuadratic`. This is because solvers and optimizers should rely on different choices of this constant.
"""
const DEFAULT_BIERLAIRE_ε::Float64 = 2eps(Float32)

"""
    DEFAULT_BIERLAIRE_ξ

A constant on basis of which the `b` in [`BierlaireQuadraticState`](@ref) is perturbed in order "to avoid stalling" (see [bierlaire2015optimization; Chapter 11.2.1](@cite); in this reference the author recommends ``10^{-7}`` as a value).
Its value is $(DEFAULT_BIERLAIRE_ξ).

!!! warn
    We have deactivated the use of this constant for the moment and are only using `eps(T)` in `BierlaireQuadratic`. This is because solvers and optimizers should rely on different choices of this constant.
"""
const DEFAULT_BIERLAIRE_ξ::Float64 = 2eps(Float32)

"""
    default_precision(T)

Compute the default precision used for [`BierlaireQuadraticState`](@ref).
Compare this to the [`default_tolerance`](@ref) used in [`Options`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_precision)
default_precision(Float64)

# output

2.220446049250313e-16
```

```jldoctest; setup = :(using SimpleSolvers: default_precision)
default_precision(Float32)

# output

1.1920929f-6
```

```jldoctest; setup = :(using SimpleSolvers: default_precision)
default_precision(Float16)

# output

ERROR: No default precision defined for Float16.
[...]
```
"""
default_precision

function default_precision(::Type{Float32})
    10eps(Float32)
end

function default_precision(::Type{Float64})
    eps(Float64)
end

function default_precision(::Type{T}) where {T <: AbstractFloat}
    error("No default precision defined for $(T).")
end

"""
    BierlaireQuadraticState <: LinesearchState


# Extended help

Note that the performance of [`BierlaireQuadratic`](@ref) may heavily depend on the choice of [`DEFAULT_BIERLAIRE_ε`](@ref) (i.e. the precision) and [`DEFAULT_BIERLAIRE_ξ`](@ref).
"""
struct BierlaireQuadraticState{T} <: LinesearchState{T}
    config::Options{T}

    ε::T
    ξ::T

    function BierlaireQuadraticState(T₁::DataType=Float64;
                    ε::T = default_precision(T₁), # DEFAULT_BIERLAIRE_ε,
                    ξ::T = default_precision(T₁), # DEFAULT_BIERLAIRE_ξ,
                    options_kwargs...) where {T}
        config₁ = Options(T₁; options_kwargs...)
        new{T₁}(config₁, T₁(ε), T₁(ξ))
    end
end

LinesearchState(algorithm::BierlaireQuadratic; T::DataType=Float64, kwargs...) = BierlaireQuadraticState(T; kwargs...)

function (ls::BierlaireQuadraticState{T})(obj::AbstractUnivariateObjective{T}, a::T, b::T, c::T, iteration_number::Integer=1) where {T}
    ls(obj.F, a, b, c, iteration_number)
end

"""
    shift_χ_to_avoid_stalling(χ, a, b, c, ε)

Check whether `b` is closer to `a` or `c` and shift `χ` accordingly.
"""
function shift_χ_to_avoid_stalling(χ::T, a::T, b::T, c::T, ε::T) where {T}
    if (c - b) > (b - a)
        χ + ε / 2
    else
        χ - ε / 2
    end
end

function (ls::BierlaireQuadraticState{T})(fˡˢ::Callable, a::T, b::T, c::T, iteration_number::Integer) where {T}
    (iteration_number != MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH) || return b
    χ = T(.5) * ( fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2) ) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
    # perform a perturbation if χ ≈ b (in order "to avoid stalling")
    χ = b == χ ? shift_χ_to_avoid_stalling(χ, a, b, c, ls.ε) : χ
    if χ > b
        if fˡˢ(χ) > fˡˢ(b)
            c = χ
        else
            a, b = b, χ
        end
    else
        if fˡˢ(χ) > fˡˢ(b)
            a = χ
        else
            c, b = b, χ
        end
    end
    !( ((c - a) ≤ ls.ε) ) || !( ((fˡˢ(a) - fˡˢ(b)) ≤ ls.ε) && ((fˡˢ(c) - fˡˢ(b)) ≤ ls.ε) ) || return b
    # ( (c - a) ≤ ls.ε ) || return b
    ls(fˡˢ, a, b, c, iteration_number+1)
end

function (ls::BierlaireQuadraticState{T})(f, x₀::T=T(0.0), iteration_number::Integer=1) where {T}
    # check if the minimum has already been reached
    !(l2norm(derivative(f, x₀)) < ls.ξ) || return x₀
    ls(f, triple_point_finder(f, x₀)..., iteration_number)
end