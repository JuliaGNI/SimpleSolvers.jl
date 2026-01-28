"""
    DEFAULT_BIERLAIRE_ε

A constant that determines the *precision* in [`BierlaireQuadratic`](@ref). The constant recommended in [bierlaire2015optimization](@cite) is `1E-3`.

Note that this constant may also depend on whether we deal with optimizers or solvers.

!!! warning
    We have deactivated the use of this constant for the moment and are only using `eps(T)` in `BierlaireQuadratic`. This is because solvers and optimizers should rely on different choices of this constant.
"""
const DEFAULT_BIERLAIRE_ε::Float64 = 2eps(Float32)

"""
    DEFAULT_BIERLAIRE_ξ

A constant on basis of which the `b` in [`BierlaireQuadratic`](@ref) is perturbed in order "to avoid stalling" (see [bierlaire2015optimization; Chapter 11.2.1](@cite); in this reference the author recommends ``10^{-7}`` as a value).
Its value is $(DEFAULT_BIERLAIRE_ξ).

!!! warning
    We have deactivated the use of this constant for the moment and are only using `eps(T)` in `BierlaireQuadratic`. This is because solvers and optimizers should rely on different choices of this constant.
"""
const DEFAULT_BIERLAIRE_ξ::Float64 = 2eps(Float32)

"""
    default_precision(T)

Compute the default precision used for [`BierlaireQuadratic`](@ref).
Compare this to the [`default_tolerance`](@ref) used in [`Options`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_precision)
default_precision(Float64)

# output

1.7763568394002505e-15
```

```jldoctest; setup = :(using SimpleSolvers: default_precision)
default_precision(Float32)

# output

9.536743f-7
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
    8eps(Float32)
end

function default_precision(::Type{Float64})
    8eps(Float64)
end

function default_precision(::Type{T}) where {T<:AbstractFloat}
    error("No default precision defined for $(T).")
end

"""
    BierlaireQuadratic <: Linesearch

Algorithm taken from [bierlaire2015optimization](@cite).

# Extended help

Note that the performance of [`BierlaireQuadratic`](@ref) may heavily depend on the choice of [`DEFAULT_BIERLAIRE_ε`](@ref) (i.e. the precision) and [`DEFAULT_BIERLAIRE_ξ`](@ref).
"""
struct BierlaireQuadratic{T} <: LinesearchMethod{T}
    ε::T
    ξ::T

    function BierlaireQuadratic(T₁::DataType=Float64;
        ε::T=default_precision(T₁), # DEFAULT_BIERLAIRE_ε,
        ξ::T=default_precision(T₁) # DEFAULT_BIERLAIRE_ξ
        ) where {T}
        new{T₁}(T₁(ε), T₁(ξ))
    end
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, a::T, b::T, c::T, iteration_number::Integer=1) where {T, LST <: BierlaireQuadratic}
    bierlaire_quadratic(problem.F, ls, a, b, c, iteration_number)
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, x₀::T=zero(T), iteration_number::Integer=1) where {T, LST <: BierlaireQuadratic}
    # check if the minimum has already been reached
    !(l2norm(derivative(problem, x₀)) < ls.algorithm.ξ) || return x₀
    solve(problem, ls, triple_point_finder(problem, x₀)..., iteration_number)
end


Base.show(io::IO, ls::BierlaireQuadratic) = print(io, "Bierlaire Quadratic with ε = " * string(ls.ε) * ", and ξ = " * string(ls.ξ) * ".")

"""
    shift_χ_to_avoid_stalling(χ, a, b, c, ε)

Check whether `b` is closer to `a` or `c` and shift `χ` accordingly. This is taken from [bierlaire2015optimization](@cite).
"""
function shift_χ_to_avoid_stalling(χ::T, a::T, b::T, c::T, ε::T) where {T}
    if (c - b) > (b - a)
        χ + ε / 2
    else
        χ - ε / 2
    end
end

function bierlaire_quadratic(fˡˢ::Callable, ls::Linesearch{T, LST}, a::T, b::T, c::T, iteration_number::Integer) where {T, LST <: BierlaireQuadratic{T}}
    (iteration_number != max_number_of_quadratic_linesearch_iterations(T)) ||
        ((ls.config.verbosity >= 2 && @warn "Maximum number of iterations was reached."); return b)
    χ = T(0.5) * (fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2)) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
    # perform a perturbation if χ ≈ b (in order "to avoid stalling")
    χ = b == χ ? shift_χ_to_avoid_stalling(χ, a, b, c, ls.algorithm.ε) : χ
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
    !(((c - a) ≤ ls.algorithm.ε)) || !(((fˡˢ(a) - fˡˢ(b)) ≤ ls.algorithm.ε) && ((fˡˢ(c) - fˡˢ(b)) ≤ ls.algorithm.ε)) || return b
    # ( (c - a) ≤ ls.ε ) || return b
    bierlaire_quadratic(fˡˢ, ls, a, b, c, iteration_number + 1)
end

function Base.convert(::Type{T}, algorithm::BierlaireQuadratic) where {T}
    T ≠ eltype(algorithm) || return algorithm
    if algorithm.ε == default_precision(eltype(algorithm)) && algorithm.ξ == default_precision(eltype(algorithm))
        BierlaireQuadratic(T; ε=default_precision(T), ξ =default_precision(T))
    else
        BierlaireQuadratic(T; ε=T(algorithm.ε), ξ=T(algorithm.ξ))
    end
end