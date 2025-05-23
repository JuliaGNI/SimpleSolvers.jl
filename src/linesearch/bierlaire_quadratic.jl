"""
    DEFAULT_BIERLAIRE_ε

A constant that determines the *precision* in [`BierlaireQuadraticState`](@ref). The constant recommended in [bierlaire2015optimization](@cite) is `1E-3`.

Note that this constant may also depend on whether we deal with optimizers or solvers.
"""
const DEFAULT_BIERLAIRE_ε::Float64 = eps(Float32)

"""
    DEFAULT_BIERLAIRE_ξ

A constant on basis of which the `b` in [`BierlaireQuadraticState`](@ref) is perturbed in order "to avoid stalling" (see [bierlaire2015optimization; Chapter 11.2.1](@cite)).
Its value is $(DEFAULT_BIERLAIRE_ξ).
"""
const DEFAULT_BIERLAIRE_ξ = 1E-7

"""
    BierlaireQuadraticState <: LinesearchState


# Extended help

Note that the performance of [`BierlaireQuadratic`](@ref) may heavily depend on the choice of [`DEFAULT_BIERLAIRE_ε`](@ref) (i.e. the precision) and [`DEFAULT_BIERLAIRE_ξ`](@ref).
"""
struct BierlaireQuadraticState{T,OPT} <: LinesearchState where {T <: Number, OPT <: Options{T}}
    config::OPT

    ε::T
    ξ::T

    function BierlaireQuadraticState(T₁::DataType=Float64; config = Options(),
                    ε::T = DEFAULT_BIERLAIRE_ε,
                    ξ::T = DEFAULT_BIERLAIRE_ξ) where {T}
        config₁ = Options(T₁, config)
        new{T₁, typeof(config₁)}(config₁, T₁(ε), T₁(ξ))
    end
end

LinesearchState(algorithm::BierlaireQuadratic; T::DataType=Float64, kwargs...) = BierlaireQuadraticState(T; kwargs...)

function (ls::BierlaireQuadraticState{T})(obj::AbstractUnivariateObjective{T}, a::T, b::T, c::T) where {T}
    ls(obj.F, a, b, c)
end

function (ls::BierlaireQuadraticState{T})(f::Callable, a::T, b::T, c::T) where {T}
    for _ in ls.config.max_iterations
        χ = T(.5) * ( f(a) * (b^2 - c^2) + f(b) * (c^2 - a^2) + f(c) * (a^2 - b^2) ) / (f(a) * (b - c) + f(b) * (c - a) + f(c) * (a - b))
        # perform a perturbation if χ ≈ b (in order "to avoid stalling")
        χ = (χ ≈ b) ? χ * (one(T) + ls.ξ) : χ
        χ = if (c - b) > (b - a)
            χ + ls.ε / 2
        else
            χ - ls.ε / 2
        end
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
                b, c = χ, b
            end
        end
        if ( (f(a) ≤ ls.ε) && ((f(c) - f(b)) ≤ ls.ε) ) || ( (c - a) ≤ ls.ε )
            break
        end
    end
    b
end

function (ls::BierlaireQuadraticState{T})(f, x₀::T=T(0.0)) where {T}
    # check if the minimum has already been reached
    !(l2norm(derivative(f, x₀)) < ls.ξ) || return x₀
    ls(f, triple_point_finder(f, x₀)...)
end