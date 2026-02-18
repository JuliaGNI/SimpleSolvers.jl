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
function default_precision end

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


"""
    BierlaireQuadratic <: Linesearch

Algorithm taken from [bierlaire2015optimization](@cite).

# Extended help

Note that the performance of [`BierlaireQuadratic`](@ref) may heavily depend on the choice of [`DEFAULT_BIERLAIRE_ε`](@ref) (i.e. the precision) and [`DEFAULT_BIERLAIRE_ξ`](@ref).
"""
struct BierlaireQuadratic{T} <: LinesearchMethod{T}
    ε::T
    ξ::T

    function BierlaireQuadratic{T}(ε::T, ξ::T) where {T}
        new{T}(ε, ξ)
    end
end

function BierlaireQuadratic(::Type{T}=Float64;
    ε=default_precision(T), # DEFAULT_BIERLAIRE_ε,
    ξ=default_precision(T)  # DEFAULT_BIERLAIRE_ξ
) where {T}
    BierlaireQuadratic{T}(ε, ξ)
end

BierlaireQuadratic(::Type{T}, ::SolverMethod) where {T} = BierlaireQuadratic(T)

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, a::T, b::T, c::T, params, iteration_number::Integer) where {T}
    f = x -> problem(ls).F(x, params)
    (iteration_number != max_number_of_quadratic_linesearch_iterations(T)) ||
        ((ls.config.verbosity >= 2 && @warn "Maximum number of iterations was reached."); return b)
    χ = T(0.5) * (f(a) * (b^2 - c^2) + f(b) * (c^2 - a^2) + f(c) * (a^2 - b^2)) / (f(a) * (b - c) + f(b) * (c - a) + f(c) * (a - b))
    # perform a perturbation if χ ≈ b (in order "to avoid stalling")
    χ = b == χ ? shift_χ_to_avoid_stalling(χ, a, b, c, method(ls).ε) : χ
    if χ > b
        if f(χ) > f(b)
            c = χ
        else
            a, b = b, χ
        end
    else
        if f(χ) > f(b)
            a = χ
        else
            c, b = b, χ
        end
    end
    !(((c - a) ≤ method(ls).ε)) || !(((f(a) - f(b)) ≤ method(ls).ε) && ((f(c) - f(b)) ≤ method(ls).ε)) || return b
    # ( (c - a) ≤ ls.ε ) || return b
    solve(ls, a, b, c, params, iteration_number + 1)
end

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, α₀::T, params, iteration_number::Integer) where {T}
    # check if the minimum has already been reached
    !(l2norm(derivative(problem(ls), α₀, params)) < method(ls).ξ) || return α₀
    solve(ls, triple_point_finder(problem(ls), params, α₀)..., params, iteration_number)
end

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, α₀::T, params=NullParameters()) where {T}
    # TODO: The following line should use α₀ instead of zero(T) but that requires a rework of the bracketing algorithm
    # solve(problem, ls, α₀, params, 1)
    solve(ls, zero(T), params, 1)
end



Base.show(io::IO, ls::BierlaireQuadratic) = print(io, "Bierlaire Quadratic with ε = " * string(ls.ε) * ", and ξ = " * string(ls.ξ) * ".")

function Base.convert(::Type{T}, method::BierlaireQuadratic{AT}) where {T,AT}
    T ≠ AT || return method
    if method.ε == default_precision(AT) && method.ξ == default_precision(AT)
        BierlaireQuadratic{T}(default_precision(T), default_precision(T))
    else
        BierlaireQuadratic{T}(T(method.ε), T(method.ξ))
    end
end

function Base.isapprox(bq₁::BierlaireQuadratic{T}, bq₂::BierlaireQuadratic{T}; kwargs...) where {T}
    isapprox(bq₁.ε, bq₂.ε; kwargs...) && isapprox(bq₁.ξ, bq₂.ξ; kwargs...)
end
